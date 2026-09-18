"""
Pure-NumPy port of the CoLA benchmark: freeze the pretrained byte model,
carry its recurrent state across the raw bytes of each sentence, then train
a tiny linear head on the final state of the last layer.

Ports 1:1 from the MLX script:
  - the backbone is frozen: only the head's two arrays get gradients
  - `dummies` are created once per epoch and never touched (always zero) --
    kept for parity even though nothing depends on them, since it means the
    recurrent update reduces to "advance the trace with input only".
  - layer.states advances every byte across the WHOLE dataset, never reset
    between epochs or between sentences. That's a property of the original
    script, not something introduced here -- see the note at the bottom of
    this file.
"""

import math
import numpy as np

from llm_main import AdamW, Model


# --------------------------------------------------------------------------
# classification head
# --------------------------------------------------------------------------

class Classification:
    """Linear(dim, 2), hand-rolled forward/backward."""

    def __init__(self, dim, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        k = 1.0 / math.sqrt(dim)          # mlx.nn.Linear init: U(-k, k)
        self.p = {
            "proj.weight": rng.uniform(-k, k, (2, dim)).astype(dtype),
            "proj.bias": rng.uniform(-k, k, (2,)).astype(dtype),
        }

    def forward(self, x):
        return self.p["proj.weight"] @ x + self.p["proj.bias"]

    def backward(self, x, logits, target):
        """Softmax cross-entropy. Returns (loss, grads)."""
        m = logits.max()
        e = np.exp(logits - m)
        probs = e / e.sum()

        loss = float(-np.log(probs[target] + 1e-12))

        dlogits = probs.copy()
        dlogits[target] -= 1.0

        grads = {
            "proj.weight": np.outer(dlogits, x),
            "proj.bias": dlogits,
        }
        return loss, grads


# --------------------------------------------------------------------------
# data + metric
# --------------------------------------------------------------------------

def cola(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                data.append((parts[3].encode("utf-8"), int(parts[1])))
    return data


def mcc(tp, tn, fp, fn):
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator != 0.0 else 0.0


# --------------------------------------------------------------------------
# frozen-backbone forward
# --------------------------------------------------------------------------

def encode_bytes(model, byte_seq):
    """Run the frozen model over one byte string, carrying model.buf state
    forward byte-by-byte (no gradient, no eligibility-trace update -- exactly
    what the MLX loop did: forward, then `layer.states = stop_gradient(state)`,
    nothing else).

    Returns the final state of the LAST layer, matching
    `final = model.layers[-1].states` in the original.
    """
    zero_dummies = [np.zeros(model.dim, model.dtype) for _ in range(model.L)]

    final = None
    for b in byte_seq:
        _, _, cache = model.forward(int(b), zero_dummies)
        for i in range(model.L):
            model.buf[f"states.{i}"] = cache["state"][i].astype(model.dtype)
        final = model.buf[f"states.{model.L - 1}"]

    return final


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def run(weights_path="smaller-4.5m.safetensors", cola_path="CoLA/original/raw/in_domain_train.tsv"):
    model = Model(dim=512, layers=16, temp=0.75, lr=5e-4)
    if weights_path.endswith(".safetensors"):
        model.load_mlx_safetensors(weights_path)
    else:
        model.load(weights_path)
    # frozen: we simply never call model.opt.update, only head.opt does

    head = Classification(model.dim, dtype=model.dtype)
    headopt = AdamW(lr=1e-3)   # matches opt.AdamW(learning_rate=1e-3) in the original

    data = cola(cola_path)

    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}")

        tp = tn = fp = fn = 0
        score = 0.0

        for i, (byte_seq, label) in enumerate(data):
            final = encode_bytes(model, byte_seq)

            logits = head.forward(final)
            loss, grads = head.backward(final, logits, label)
            headopt.update(head.p, grads)

            predicted_class = int(np.argmax(logits))
            if predicted_class == 1 and label == 1:
                tp += 1
            elif predicted_class == 0 and label == 0:
                tn += 1
            elif predicted_class == 1 and label == 0:
                fp += 1
            elif predicted_class == 0 and label == 1:
                fn += 1

            score = mcc(tp, tn, fp, fn)

            if i > 0 and i % 500 == 0:
                print(f"{i + 1}: TP, TN, FP, FN | {tp}, {tn}, {fp}, {fn} ({score})")

        print(f"{i + 1}: TP, TN, FP, FN | {tp}, {tn}, {fp}, {fn} ({score})")


if __name__ == "__main__":
    run()

# --------------------------------------------------------------------------
# NOTE on state carry, ported unchanged from the MLX version
# --------------------------------------------------------------------------
# `model.buf[f"states.{i}"]` is advanced inside encode_bytes() and never
# reset -- not between sentences, not between epochs. So sentence 5000's
# starting hidden state is whatever sentence 4999 left behind, and by epoch 2
# it's whatever the end of epoch 1 left behind. If that's intentional (you
# want the probe to reflect the model's behaviour as a continually-running
# stream, matching how it's used in chat mode) leave this alone. If you
# actually want each sentence encoded independently -- which is the usual
# setup for a linear probe, and removes a confound where CoLA's row order
# leaks into the "representation" being scored -- reset the states at the
# top of encode_bytes():
#
#   def encode_bytes(model, byte_seq):
#       for i in range(model.L):
#           model.buf[f"states.{i}"] = np.zeros(model.dim, model.dtype)
#       ...
#
# I left it exactly as the original behaves; this is a one-line change if
# you decide the leak is unwanted.