"""
Pure-NumPy port of the MLX continual-learning byte model.

Everything that MLX did implicitly (autodiff, module parameter registration,
AdamW) is written out by hand here so that every gradient is an inspectable
NumPy array.

Structure
---------
    forward(c, dummies)            -> output logits, stop prob, cache
    backward(cache, n, end)        -> loss, grads, dlds
    apply_traces(grads, cache, dlds)  -> eligibility-trace surgery + buffer advance
    step(c, n, end)                -> the full training step (all of the above + AdamW)

`grads` is a flat dict {name: ndarray} so you can print, clip, zero, or swap
any single gradient without fighting a module tree.

Key naming (flat, mirrors the MLX tree so checkpoints map 1:1):
    encoder.embed.weight        (256, dim)
    decoder.decode.weight       (256, dim)      decoder.decode.bias   (256,)
    decoder.stop.weight         (1, dim)        decoder.stop.bias     (1,)
    layers.{i}.decay            (dim,)
    layers.{i}.norm.weight      (dim,)          layers.{i}.norm.bias  (dim,)
    layers.{i}.weights.weight   (dim, dim)      # y = W @ x, W is (out, in)

Buffers (recurrent state + eligibility traces, NOT optimized):
    embedtrace                  (256, dim)
    states.{i}                  (dim,)
    decaytrace.{i}              (dim,)


DIFFERENCES FROM THE MLX ORIGINAL  (read this before comparing runs)
--------------------------------------------------------------------
1. In MLX, *any* mx.array attribute of an nn.Module is registered as a
   parameter. That means `self.states`, `self.decaytrace` and
   `self.embedtrace` were all inside `trainable_parameters()`, so
   `optimizer.update(self, grads)` was applying AdamW steps to the recurrent
   state and to the eligibility traces themselves, right after they had been
   explicitly assigned. That is almost certainly unintended. Here they live in
   `self.buf` and the optimizer never touches them.
   Set OPTIMIZE_BUFFERS = True to reproduce the MLX behaviour bit-for-bit.

2. MLX's Adam/AdamW do not apply bias correction by default. `AdamW` below
   matches that (bias_correction=False). Flip it on to get textbook Adam.

3. `mx.var` is population variance (ddof=0); `np.var` default matches.

4. Checkpoints are .npz here. `load_mlx_safetensors()` converts an existing
   MLX .safetensors checkpoint (weight layouts are identical).
"""


"""
Pure-NumPy port of the MLX continual-learning byte model.

Everything that MLX did implicitly (autodiff, module parameter registration,
AdamW) is written out by hand here so that every gradient is an inspectable
NumPy array.

Structure
---------
    forward(c, dummies)            -> output logits, stop prob, cache
    backward(cache, n, end)        -> loss, grads, dlds
    apply_traces(grads, cache, dlds)  -> eligibility-trace surgery + buffer advance
    step(c, n, end)                -> the full training step (all of the above + AdamW)

`grads` is a flat dict {name: ndarray} so you can print, clip, zero, or swap
any single gradient without fighting a module tree.

Key naming (flat, mirrors the MLX tree so checkpoints map 1:1):
    encoder.embed.weight        (256, dim)
    decoder.decode.weight       (256, dim)      decoder.decode.bias   (256,)
    decoder.stop.weight         (1, dim)        decoder.stop.bias     (1,)
    layers.{i}.decay            (dim,)
    layers.{i}.norm.weight      (dim,)          layers.{i}.norm.bias  (dim,)
    layers.{i}.weights.weight   (dim, dim)      # y = W @ x, W is (out, in)

Buffers (recurrent state + eligibility traces, NOT optimized):
    embedtrace                  (256, dim)
    states.{i}                  (dim,)
    decaytrace.{i}              (dim,)


DIFFERENCES FROM THE MLX ORIGINAL  (read this before comparing runs)
--------------------------------------------------------------------
1. In MLX, *any* mx.array attribute of an nn.Module is registered as a
   parameter. That means `self.states`, `self.decaytrace` and
   `self.embedtrace` were all inside `trainable_parameters()`, so
   `optimizer.update(self, grads)` was applying AdamW steps to the recurrent
   state and to the eligibility traces themselves, right after they had been
   explicitly assigned. That is almost certainly unintended. Here they live in
   `self.buf` and the optimizer never touches them.
   Set OPTIMIZE_BUFFERS = True to reproduce the MLX behaviour bit-for-bit.

2. MLX's Adam/AdamW do not apply bias correction by default. `AdamW` below
   matches that (bias_correction=False). Flip it on to get textbook Adam.

3. `mx.var` is population variance (ddof=0); `np.var` default matches.

4. Checkpoints are .npz here. `load_mlx_safetensors()` converts an existing
   MLX .safetensors checkpoint (weight layouts are identical).
"""


import glob
import itertools
import math
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm

import numpy as np

DTYPE = np.float32
LN_EPS = 1e-5          # mlx.nn.LayerNorm default
OPTIMIZE_BUFFERS = False   # see note 1 above
_OPT_AVAILABLE = False


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def sigmoid(x):
    # numerically stable, no overflow warnings on large |x|
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def silu(x):
    return x * sigmoid(x)


def dsilu(x):
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def logsumexp(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def layernorm_fwd(x, g, b, eps=LN_EPS):
    mu = x.mean()
    var = x.var()
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    return xhat * g + b, xhat, rstd


def layernorm_bwd(dy, xhat, rstd, g):
    """Returns dx, dgamma, dbeta for a 1-D LayerNorm over the whole vector."""
    dg = dy * xhat
    db = dy
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean() - xhat * (dxhat * xhat).mean())
    return dx, dg, db


# --------------------------------------------------------------------------
# optimizer
# --------------------------------------------------------------------------

class AdamW:
    """Decoupled weight decay, MLX-compatible update order.

    MLX does:  w <- w * (1 - lr * wd)   then the Adam step, and by default it
    skips bias correction entirely.
    """

    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01,
                 bias_correction=False):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.wd = weight_decay
        self.bias_correction = bias_correction
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for k, g in grads.items():
            if k not in params:
                continue
            w = params[k]
            if k not in self.m:
                self.m[k] = np.zeros_like(w)
                self.v[k] = np.zeros_like(w)

            if self.wd:
                w = w * (1.0 - self.lr * self.wd)

            m = self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            v = self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * (g * g)

            if self.bias_correction:
                m = m / (1.0 - self.b1 ** self.t)
                v = v / (1.0 - self.b2 ** self.t)

            params[k] = (w - self.lr * m / (np.sqrt(v) + self.eps)).astype(w.dtype)

    def state_dict(self):
        d = {f"o.m.{k}": v for k, v in self.m.items()}
        d.update({f"o.v.{k}": v for k, v in self.v.items()})
        d["o.t"] = np.array(self.t)
        return d

    def load_state_dict(self, d):
        for k, v in d.items():
            if k.startswith("o.m."):
                self.m[k[4:]] = v
            elif k.startswith("o.v."):
                self.v[k[4:]] = v
            elif k == "o.t":
                self.t = int(v)


class WeightEncoding:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
    

    def eigenvalue_encoder(self, x):
        eps = 1e-5
        raw_X = np.asarray(x)
        AME = self.AME_Encoder(raw_X)  
        AMR = 1.0 / (1.0 + np.exp(-AME)) + eps
        mag = np.mean(np.linalg.norm(raw_X, axis=-1))

        if raw_X.ndim > 2:
            raw_X = raw_X.reshape(raw_X.shape[0], -1)

        anisotropy = self.anisotropy_measurement(raw_X)

        structured_noise = np.random.uniform(0, mag, size=raw_X.shape)
        X = np.vstack((raw_X, structured_noise))
        if X.ndim == 2 and X.shape[1] == 1:
            X = np.hstack((raw_X, structured_noise))

        cov = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]

        energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        energy_sigmoid_growth = 1.0 / (1.0 + np.exp(-energy))
        energy_consistency = np.std(energy_sigmoid_growth)
        k = np.searchsorted(energy, 0.90) + 1     # +1 converts 0-based index to count

        trA = k / (1.0 - anisotropy) + eps  
        trB = (1/2 + energy_consistency) / (1.0 + trA**2)
        trC = (1/6 + AMR) / (1.0 - trB**2) + eps

        if np.isnan(trC) or np.isinf(trC):
            trC = anisotropy * (trB**2 - 1.0) + eps
            if np.isnan(trC) or np.isinf(trC):
                trC = (1.0 - AMR)

        min_val = min(trC, 0) 
        max_val = max(trC, 0) 
        floating_point = np.random.uniform(min_val, max_val, size=X.shape) 
        return k, floating_point, structured_noise


    def spectral_signature(self, x, structured_noise, k=5):
        raw_X = np.asarray(x, dtype=np.float64)
        if raw_X.ndim > 2:
            X = raw_X.reshape(raw_X.shape[0], -1)
        else:
            X = raw_X.reshape(raw_X.shape[0], -1)

        X = np.atleast_2d(X)

        if X.ndim == 2 and X.shape[1] == 1:
            # normalize structured_noise to 2D matching X's row count
            noise = np.asarray(structured_noise, dtype=np.float64)

            if noise.ndim == 1:
                # reshape to (n_samples, n_noise_features)
                # if noise length matches X's row count, treat as column vector
                if noise.shape[0] == X.shape[0]:
                    noise = noise.reshape(-1, 1)
                else:
                    # noise is a flat feature vector not aligned to X's rows —
                    # broadcast it across all rows instead of stacking blindly
                    noise = np.tile(noise.reshape(1, -1), (X.shape[0], 1))
            elif noise.ndim > 2:
                noise = noise.reshape(noise.shape[0], -1)

            # align row counts before hstack
            if noise.shape[0] != X.shape[0]:
                min_rows = min(noise.shape[0], X.shape[0])
                X     = X[:min_rows]
                noise = noise[:min_rows]
                print(f'[⚠️] spectral_signature: row count mismatch, '
                    f'truncated to {min_rows} rows')

            X = np.hstack((X, noise))

        # guard against degenerate covariance — need at least 2 samples
        if X.shape[0] < 2:
            print(f'[⚠️] spectral_signature: only {X.shape[0]} sample(s), '
                f'cannot compute covariance — returning zeros')
            return np.zeros(k)

        try:
            cov     = np.cov(X, rowvar=False, ddof=1)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
            eig_sum = eigvals.sum()
            if eig_sum <= 1e-8:
                return np.zeros(min(k, len(eigvals)))
            return eigvals[:k] / (eig_sum + 1e-8)
        except np.linalg.LinAlgError as e:
            print(f'[⚠️] spectral_signature: eigendecomposition failed: {e}')
            return np.zeros(k)


    def spectral_similarity(self, a, b, structured_noise):
        sa = self.spectral_signature(a, structured_noise)
        sb = self.spectral_signature(b, structured_noise)
        if sa.shape != sb.shape:
            min_rows = min(sa.shape[0], sb.shape[0])

            sa = sa[:min_rows]
            sb = sb[:min_rows]

        return np.exp(-np.linalg.norm(sa - sb))

    # abstract modelling error provides the model how to better process weights when the data complexity has little geometric complexity
    def AME_Encoder(self, x):
        X = np.asarray(x)

        if len(X) == 0:
            print('[!] X size is 0, AME Will be replaced by minimum confidence threshold')
            return 0.0

        if _OPT_AVAILABLE and np.asarray(X).ndim == 2:
            return optimized_ame_encoder(np.asarray(X, dtype=np.float64))     

        try:
            gradient = np.gradient(x)
        except:
            subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
            gradient = np.gradient(subnet.flatten())

        mean_vector_mag  = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))
        # Regular AME Equations, higher AME provides capabilities for the model to experience errors during abstraction
        # Lower AME means lower chance for un optimal abstraction.

        AME =  np.log1p(X_mag) * np.log1p(mean_vector_mag) 
        return AME

    # anisotropy provides the model the standard complexity of the data geometry, allowing it to know how complex the data needs to be processed.
    def anisotropy_measurement(self, x):
        eps = 1e-5
        if _OPT_AVAILABLE:
            x = np.asarray(x)            
            x = x.reshape(x.shape[0], -1)
            return optimized_anisotropy(np.asarray(x, dtype=np.float64))

        try:
            gradient = np.gradient(x)
        except:
            subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
            gradient = np.gradient(subnet.flatten())

        val = [np.linalg.norm(v) for v in gradient]
        anisotropy = np.std(val) / np.mean(val) + eps

        return anisotropy

    # weight shaping provides directional context in which how the data should be processed in order to align with the data geometry
    def abstract_weight_shaping(self, x, seed=0):
        input_size = self.input_size
        output_size = self.output_size

        eps = 1e-5
        x = np.asarray(x)

        rng = np.random.default_rng(seed)

        anisotropy = self.anisotropy_measurement(x)
        mag = np.mean(np.linalg.norm(x))

        k, floating_point, structured_noise = self.eigenvalue_encoder(x)
        AME = self.AME_Encoder(x)
        AMR = 1.0 / (1.0 + np.exp(-AME)) # abstract modelling rate        

        spectral_similarity = self.spectral_similarity(x, floating_point, structured_noise)

        AEL = (0.3 + spectral_similarity + eps) * anisotropy 
        scaled_anisotropy = anisotropy / (anisotropy + 1.0)
        
        abstraction_efficiency = (1.0 + AEL) * (1.0 - AMR) + eps
        abstraction_efficiency = (k + AEL) * (1.0 - AMR) + eps

        if np.isnan(abstraction_efficiency) or np.isinf(abstraction_efficiency):
            abstraction_efficiency = (1 - AMR) + eps

        abstract_context = rng.uniform(-abstraction_efficiency, abstraction_efficiency, size=(input_size, output_size)) 

        return abstract_context



    def weight_shaping(self, x, type=None):
        if np.isnan(x).any() or np.isinf(x).any():
            x = np.nan_to_num(x, nan=0.0, posinf=1e99, neginf=-1e99)     

        if isinstance(x, list):
            x = np.asarray(x)

        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        if np.std(x) == 0:
            x = np.random.uniform(0, 1, size=x.shape)

        abstract_context = self.abstract_weight_shaping(x)
        abstract_context /= np.max(np.abs(abstract_context)) + 1e-5  # Normalized to [-1, 1]

        return abstract_context



# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class Model:
    def __init__(self, dim, layers, temp, lr, weight_decay=0.01,
                 seed=0, dtype=DTYPE):
        self.dim = dim
        self.L = layers
        self.temp = temp
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)
        self.layers = layers

        rng = self.rng
        emb_scale = 1.0 / math.sqrt(dim)      # mlx.nn.Embedding init
        k = 1.0 / math.sqrt(dim)              # mlx.nn.Linear init: U(-k, k)

        p = {}
        self.p = p
        self.weight_encoding = None

        buf = {"embedtrace": np.zeros((256, dim), dtype)}
        for i in range(layers):
            buf[f"states.{i}"] = np.zeros(dim, dtype)
            buf[f"decaytrace.{i}"] = np.zeros(dim, dtype)
        self.buf = buf

        self.opt = AdamW(lr, weight_decay=weight_decay)

    # ---------------- forward ----------------

    def forward(self, c, dummies=None):
        """One byte through the network. Returns (output, stop, cache).

        `dummies` is the NumPy stand-in for the MLX zero-tensor trick: each
        entry is added straight into that layer's state, so dL/ddummy_i is
        exactly dL/dstate_i. It is kept as an explicit argument so you can
        perturb a single layer's state and watch the effect.
        """
        if len(self.p) < 1:
            dim = self.dim
            rng = self.rng
            emb_scale = 1.0 / math.sqrt(dim)      # mlx.nn.Embedding init
            k = 1.0 / math.sqrt(dim)
            dtype = self.dtype 
            p = self.p
            layers = self.layers

            self.weight_encoding = WeightEncoding(dim, dim)
            p["encoder.embed.weight"] = rng.normal(0.0, emb_scale, (256, dim)).astype(dtype)
            x = p["encoder.embed.weight"][c]
            cache = {"c": c, "x0": x, "prev": [], "state": [], "decay": [],
                 "xhat": [], "rstd": [], "norm": [], "h": [], "xin": []}
            p["decoder.decode.weight"] = rng.uniform(-k, k, (256, dim)).astype(dtype)
            p["decoder.decode.bias"] = rng.uniform(-k, k, (256,)).astype(dtype)
            p["decoder.stop.weight"] = rng.uniform(-k, k, (1, dim)).astype(dtype)
            p["decoder.stop.bias"] = rng.uniform(-k, k, (1,)).astype(dtype)

            for i in range(layers):
                p[f"layers.{i}.decay"] = np.zeros(dim, dtype)
                p[f"layers.{i}.norm.weight"] = np.ones(dim, dtype)
                p[f"layers.{i}.norm.bias"] = np.zeros(dim, dtype)
                p[f"layers.{i}.weights.weight"] = rng.uniform(-k, k, (dim, dim)).astype(dtype) # self.weight_encoding.weight_shaping(x) 

            self.p = p

        p, buf, dt = self.p, self.buf, self.dtype
        x = p["encoder.embed.weight"][c]
        cache = {"c": c, "x0": x, "prev": [], "state": [], "decay": [],
                "xhat": [], "rstd": [], "norm": [], "h": [], "xin": []}
        if dummies is None:
            dummies = [np.zeros(self.dim, dt) for _ in range(self.L)]

        for i in range(self.L):
            decay = sigmoid(p[f"layers.{i}.decay"])
            prev = buf[f"states.{i}"]
            state = decay * prev + x + dummies[i] 

            nrm, xhat, rstd = layernorm_fwd(
                state, p[f"layers.{i}.norm.weight"], p[f"layers.{i}.norm.bias"])
            h = p[f"layers.{i}.weights.weight"] @ nrm
            s = silu(h)

            cache["xin"].append(x)
            cache["prev"].append(prev)
            cache["state"].append(state)
            cache["decay"].append(decay)
            cache["xhat"].append(xhat)
            cache["rstd"].append(rstd)
            cache["norm"].append(nrm)
            cache["h"].append(h)

            x = x + s

        cache["x"] = x
        output = p["decoder.decode.weight"] @ x + p["decoder.decode.bias"]
        stop_pre = p["decoder.stop.weight"] @ x + p["decoder.stop.bias"]
        stop = sigmoid(stop_pre)
        cache["stop_pre"] = stop_pre
        cache["stop"] = stop
        cache["output"] = output
        return output, stop, cache

    # ---------------- loss ----------------

    def loss(self, cache, n, end):
        """Scalar loss + the seed gradients dL/dx_final, dL/doutput, dL/dstop_pre.

        Terms, matching the original:
          variance loss   max(0, 1 - sqrt(var(x) + 1e-4))
          prediction MSE  mean((x - stopgrad(embed[n]))^2)
          cross-entropy   -output[n] + logsumexp(output)
          stop MSE        mean((stop - end)^2)
        """
        x = cache["x"]
        D = x.size
        dx = np.zeros_like(x)

        mu = x.mean()
        sd = np.sqrt(x.var() + 1e-4)
        total = 0.0
        parts = {}

        v = 1.0 - sd
        if v > 0.0:
            total += v
            dx += -(x - mu) / (D * sd)
        parts["variance"] = max(0.0, float(v))

        output = cache["output"]
        doutput = np.zeros_like(output)
        dstop_pre = np.zeros_like(cache["stop_pre"])

        if n is not None:
            tgt = self.p["encoder.embed.weight"][n]      # stop_gradient
            diff = x - tgt
            mse = float(np.mean(diff * diff))
            total += mse
            dx += 2.0 * diff / D
            parts["pred_mse"] = mse

            ce = float(-output[n] + logsumexp(output))
            total += ce
            doutput = softmax(output)
            doutput[n] -= 1.0
            parts["crossentropy"] = ce

            tgt_stop = np.array([1.0 if end else 0.0], dtype=x.dtype)
            sdiff = cache["stop"] - tgt_stop
            smse = float(np.mean(sdiff * sdiff))
            total += smse
            dstop = 2.0 * sdiff / sdiff.size
            dstop_pre = dstop * cache["stop"] * (1.0 - cache["stop"])
            parts["stop_mse"] = smse

        cache["loss_parts"] = parts
        return float(total), dx, doutput, dstop_pre


    # ---------------- backward ----------------

    def backward(self, cache, n, end):
        """Hand-written reverse pass.

        Returns (loss, grads, dlds) where dlds[i] == dL/dstate_i, i.e. exactly
        what MLX produced as the gradient w.r.t. the dummy tensors.

        These are the *instantaneous* (BPTT-truncated-to-one-step) gradients.
        The temporal credit assignment is bolted on afterwards in apply_traces.
        """
        p = self.p
        x = cache["x"]
        loss, dx, doutput, dstop_pre = self.loss(cache, n, end)

        g = {}

        g["decoder.decode.weight"] = np.outer(doutput, x)
        g["decoder.decode.bias"] = doutput
        dx = dx + p["decoder.decode.weight"].T @ doutput

        g["decoder.stop.weight"] = np.outer(dstop_pre, x)
        g["decoder.stop.bias"] = dstop_pre
        dx = dx + p["decoder.stop.weight"].T @ dstop_pre

        dlds = [None] * self.L 

        for i in reversed(range(self.L)):
            # x_out = x_in + silu(W @ norm(state))
            dh = dx * dsilu(cache["h"][i])
            g[f"layers.{i}.weights.weight"] = np.outer(dh, cache["norm"][i])
            dnrm = p[f"layers.{i}.weights.weight"].T @ dh

            dstate, dgamma, dbeta = layernorm_bwd(
                dnrm, cache["xhat"][i], cache["rstd"][i],
                p[f"layers.{i}.norm.weight"])
            g[f"layers.{i}.norm.weight"] = dgamma
            g[f"layers.{i}.norm.bias"] = dbeta

            dlds[i] = dstate

            # immediate decay gradient: state = sigmoid(raw) * prev + ...
            # kept for inspection; apply_traces overwrites the real entry with
            # the eligibility-trace version, which already contains this term.
            decay = cache["decay"][i]
            g[f"layers.{i}.decay"] = dstate * cache["prev"][i] * decay * (1.0 - decay)

            # residual path + the state's dependence on this layer's input x
            dx = dx + dstate

        # dx is now dL/dx0, which only touches the row that was looked up
        ge = np.zeros_like(p["encoder.embed.weight"])
        ge[cache["c"]] = dx
        g["encoder.embed.weight"] = ge

        return loss, g, dlds

    # ---------------- eligibility traces ----------------

    def apply_traces(self, g, cache, dlds):
        """Temporal credit assignment + recurrent-buffer advance.

        embedtrace_t = decay0 * embedtrace_{t-1} + onehot(c)
            d state_0 / d embed_row accumulated through the decay chain.
            The `onehot` part of the current step is already covered by the
            direct path in backward(), so only decay0 * old_trace is ADDED.

        decaytrace_t = decay * decaytrace_{t-1} + decay * (1 - decay) * prev_state
            d state / d decay_raw accumulated. The current-step term is inside
            the trace, so the decay gradient is REPLACED, not added, to avoid
            double counting.
        """
        buf = self.buf
        decay0 = cache["decay"][0]
        old_et = buf["embedtrace"]

        g["encoder.embed.weight"] = g["encoder.embed.weight"] + \
            dlds[0][None, :] * (old_et * decay0)

        onehot = (np.arange(256) == cache["c"]).astype(self.dtype)[:, None]
        buf["embedtrace"] = (old_et * decay0 + onehot).astype(self.dtype)

        for i in range(self.L):
            decay = cache["decay"][i]
            dtr = decay * buf[f"decaytrace.{i}"] + \
                decay * (1.0 - decay) * cache["prev"][i]

            g[f"layers.{i}.decay"] = dlds[i] * dtr

            buf[f"states.{i}"] = cache["state"][i].astype(self.dtype)
            buf[f"decaytrace.{i}"] = dtr.astype(self.dtype)

        if OPTIMIZE_BUFFERS:
            # reproduce the MLX quirk where buffers were registered as params
            g["embedtrace"] = dlds[0][None, :] * np.ones_like(buf["embedtrace"])
            for i in range(self.L):
                g[f"states.{i}"] = dlds[i] * cache["decay"][i]
        return g

    # ---------------- sampling ----------------

    def sample(self, output):
        probs = softmax(output)
        entropy = -np.sum(probs * np.log(probs + 1e-8)) / np.log(256.0)
        temp = max(0.1, float(self.temp * (1.0 - self.temp * entropy)))
        p = softmax(output / temp)
        p = p / p.sum()
        return int(self.rng.choice(256, p=p))

    # ---------------- the step ----------------

    def step(self, currb, nextb, end, notrace=False):
        """One byte. Mirrors Model.__call__ in the MLX version."""
        output, stop, cache = self.forward(currb)

        if notrace:
            # pure inference: no gradients, no buffer advance
            return self.sample(output), float(stop[0])

        loss, g, dlds = self.backward(cache, nextb, end)
        self.apply_traces(g, cache, dlds)

        if OPTIMIZE_BUFFERS:
            merged = dict(self.p)
            merged.update(self.buf)
            self.opt.update(merged, g)
            for k in self.p:
                self.p[k] = merged[k]
            for k in self.buf:
                self.buf[k] = merged[k]
        else:
            self.opt.update(self.p, g)

        self.last_loss = loss
        self.last_grads = g          # left on the instance for poking at
        self.last_cache = cache
        return self.sample(output), float(stop[0])

    __call__ = step

    # ---------------- checkpointing ----------------

    def save(self, path):
        data = {f"p.{k}": v for k, v in self.p.items()}
        data.update({f"b.{k}": v for k, v in self.buf.items()})
        data.update(self.opt.state_dict())
        tmp = "temporary-" + os.path.basename(path)
        tmp = os.path.join(os.path.dirname(path) or ".", tmp)
        np.savez(tmp, **data)
        os.replace(tmp + ".npz" if not tmp.endswith(".npz") else tmp, path)

    def load(self, path):
        if not os.path.exists(path):
            return
        z = np.load(path)
        opt = {}
        for k in z.files:
            v = z[k]
            if k.startswith("p."):
                self.p[k[2:]] = v.astype(self.dtype)
            elif k.startswith("b."):
                self.buf[k[2:]] = v.astype(self.dtype)
            elif k.startswith("o."):
                opt[k] = v
        self.opt.load_state_dict(opt)

    def load_mlx_safetensors(self, path):
        """Import a checkpoint written by the original MLX script.

        Weight layouts are identical (Linear is (out, in), Embedding is
        (256, dim)), so this is a pure key remap.
        """
        from safetensors.numpy import load_file
        data = load_file(path)
        for k, v in data.items():
            if k.startswith("m."):
                name = k[2:]
                if name in self.p:
                    self.p[name] = v.astype(self.dtype)
                elif name == "encoder.embedtrace":
                    self.buf["embedtrace"] = v.astype(self.dtype)
                elif name.endswith(".states"):
                    self.buf[f"states.{name.split('.')[1]}"] = v.astype(self.dtype)
                elif name.endswith(".decaytrace"):
                    self.buf[f"decaytrace.{name.split('.')[1]}"] = v.astype(self.dtype)
            elif k.startswith("o."):
                pass  # MLX optimizer tree; skip, AdamW moments restart
            elif k == "embedtrace":
                self.buf["embedtrace"] = v.astype(self.dtype)
            elif k.startswith("state."):
                self.buf[f"states.{k.split('.')[1]}"] = v.astype(self.dtype)
            elif k.startswith("decaytrace."):
                self.buf[f"decaytrace.{k.split('.')[1]}"] = v.astype(self.dtype)


# --------------------------------------------------------------------------
# gradient checking
# --------------------------------------------------------------------------

def check_gradients(model, c=65, n=66, end=False, keys=None, n_probe=6, h=1e-5):
    """Finite-difference check of the instantaneous gradients.

    Only checks what backward() claims, i.e. the within-step gradients. The
    eligibility-trace terms are temporal and are NOT covered by this: to test
    those you need a multi-step rollout with the buffers carried forward.

    Run in float64 for meaningful numbers:
        m = Model(dim=32, layers=3, temp=0.75, lr=1e-3, dtype=np.float64)
    """
    if model.dtype != np.float64:
        print("warning: run with dtype=np.float64, float32 noise swamps the check")

    saved_buf = {k: v.copy() for k, v in model.buf.items()}

    def loss_of(cc, nn, ee):
        for k, v in saved_buf.items():
            model.buf[k] = v.copy()
        _, _, cache = model.forward(cc)
        return model.loss(cache, nn, ee)[0]

    for k, v in saved_buf.items():
        model.buf[k] = v.copy()
    _, _, cache = model.forward(c)
    _, g, _ = model.backward(cache, n, end)

    rows = []
    for key in (keys or list(model.p.keys())):
        w = model.p[key]
        if key == "encoder.embed.weight":
            # every other row has an identically zero gradient, so probing at
            # random would "pass" without testing anything
            idxs = [(c, np.random.randint(0, w.shape[1])) for _ in range(n_probe)]
        else:
            idxs = [tuple(np.random.randint(0, s) for s in w.shape)
                    for _ in range(n_probe)]
        worst = 0.0
        for idx in idxs:
            orig = w[idx]
            w[idx] = orig + h
            lp = loss_of(c, n, end)
            w[idx] = orig - h
            lm = loss_of(c, n, end)
            w[idx] = orig
            num = (lp - lm) / (2 * h)
            ana = g[key][idx]
            denom = max(1e-12, abs(num) + abs(ana))
            worst = max(worst, abs(num - ana) / denom)
        rows.append((key, worst))

    for k, v in saved_buf.items():
        model.buf[k] = v

    width = max(len(r[0]) for r in rows)
    for key, rel in rows:
        flag = "ok  " if rel < 1e-5 else ("MEH " if rel < 1e-3 else "BAD ")
        print(f"{flag} {key:<{width}}  max rel err {rel:.3e}")
    return rows


def check_dlds(model, c=65, n=66, end=False, h=1e-5):
    """Verify that dlds[i] really is dL/dstate_i by perturbing the dummies."""
    saved = {k: v.copy() for k, v in model.buf.items()}
    _, _, cache = model.forward(c)
    _, _, dlds = model.backward(cache, n, end)

    for i in range(model.L):
        j = np.random.randint(model.dim)
        d = [np.zeros(model.dim, model.dtype) for _ in range(model.L)]
        d[i][j] = h
        for k, v in saved.items():
            model.buf[k] = v.copy()
        _, _, cp = model.forward(c, d)
        lp = model.loss(cp, n, end)[0]
        d[i][j] = -h
        for k, v in saved.items():
            model.buf[k] = v.copy()
        _, _, cm = model.forward(c, d)
        lm = model.loss(cm, n, end)[0]
        num = (lp - lm) / (2 * h)
        ana = dlds[i][j]
        denom = max(1e-12, abs(num) + abs(ana))
        print(f"layer {i:>2} dim {j:>4}  num {num: .6e}  ana {ana: .6e}  "
              f"rel {abs(num - ana) / denom:.3e}")
    for k, v in saved.items():
        model.buf[k] = v


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------

class Runtime:
    def __init__(self, path, threshold, **kwargs):
        self.model = Model(**kwargs)
        self.path = path
        self.threshold = threshold
        self.step_count = 0
        self.prevtime = None

    def save(self):
        self.step_count += 1
        if self.step_count % 500 == 0:
            self.model.save(self.path)

    def call(self, c, n, end, readonly=False, notrace=False):
        out = self.model.step(c, n, end, notrace)
        if not readonly:
            self.save()
        return out

    def write(self, b):
        sys.stdout.buffer.write(bytes([b]))
        sys.stdout.flush()

    def chat(self, readonly=False, notrace=False):
        while True:
            elapsed = 0 if self.prevtime is None else time.time() - self.prevtime
            text = input(f"\n[{self.now()} | {elapsed:.4f}s]\nUser >> ")
            self.prevtime = time.time()

            data = (text + "\n").encode("utf-8")
            for i, (c, n) in enumerate(itertools.pairwise(data)):
                b, _ = self.call(c, n, i == len(data) - 2, readonly, notrace)

            print(f"\n[{self.now()}]\nModel >> ", end="", flush=True)

            b = data[-1]
            while True:
                b, stop = self.call(b, None, False, readonly, notrace)
                self.write(b)
                if stop > self.threshold:
                    print()
                    break

    def dataset(self):
        files = glob.glob("wikipedia_clean/**/wiki_*", recursive=True)

        pbar = tqdm(desc="training", unit="byte", dynamic_ncols=True,
                    mininterval=0.2) if tqdm is not None else None
        while True:
            for file in files:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        data = line.encode("utf-8")
                        for i, (c, n) in enumerate(itertools.pairwise(data)):
                            b, _ = self.call(c, n, i == len(data) - 2)
                            self.write(b)

    def now(self):
        return datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

    def __call__(self):
        modes = ["train", "chat", "chatreadonly", "chatnotrace", "gradcheck"]
        try:
            mode = modes.index(input(
                f"\n'chatnotrace' runs inference with frozen state. 'chatreadonly' "
                f"chats without checkpointing. 'gradcheck' finite-differences the "
                f"backward pass.\n[{self.now()}]\nmode: {modes} >> ").lower())
        except ValueError:
            print("\nInvalid mode.")
            return

        self.model.load(self.path)
        print()

        try:
            match mode:
                case 0: self.dataset()
                case 1: self.chat()
                case 2: self.chat(readonly=True)
                case 3: self.chat(readonly=True, notrace=True)
                case 4:
                    m = Model(dim=32, layers=3, temp=0.75, lr=1e-3,
                              dtype=np.float64)
                    check_gradients(m)
                    print()
                    check_dlds(m)
        finally:
            if mode < 2:
                self.model.save(self.path)


if __name__ == "__main__":
    # Runtime(path='larger-130m.npz', threshold=0.35, dim=2048, layers=32, temp=0.75, lr=5e-4)()
    Runtime(path="smaller-4.5m.npz", threshold=0.35,
            dim=512, layers=16, temp=0.75, lr=5e-4)()