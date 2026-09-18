# Test-Model-Thing (TMT) 

[YouTube Video](https://youtu.be/9UERVVwpNew)

This is a small proof-of-concept language model (not an LLM) that incorporates the following (and some smaller features as well):
* Latent-space prediction
* Internal state + recurrent trace units (RTUs)
* Byte input/output
* Continuous data streaming
* Test-time training

The model is built with MLX, so it should run fine on all Apple Silicon devices. MLX on Linux has not been tested, but feel free to try it.

Being a proof of concept I have only trained a 4.5-million parameter model (keep in mind, GPT-1 was ~117m) for about 12 hours, but there are very promising results. The model tends to misspell characters (since it outputs byte-by-byte, rather than token-by-token) but it is able to close quotes/brackets and such. Given further training and scaling up the hyperparameters this could become much more powerful. My dataset is also tiny (only a few hundred MB), so there's a lot more world knowledge that can be fed into the model.

This model architecture was designed in about a month by me (a solo high school dev) and some Gemini (only pair programming, no agents). I wrote about a dozen prototypes before creating this architecture. I write READMEs myself without AI.

Feel free to fork the training and benchmark code (everything is under MIT). I really encourage you to try things out, submit issues, and fork the repo.

<img width="499" height="497" alt="3f7f1530-c0c7-43c4-9981-30e9023a19fb" src="https://github.com/user-attachments/assets/eb7e5a97-09b5-4a7b-9484-eb898042e9dc" />

## Training your own model

Model weights (in ```.safetensors```) are not provided because GitHub doesn't like very large files. But, you can train your own model simply by initializing a ```venv``` and installing ```mlx```, no other libraries needed, then running ```main.py```. When you run it, you will be prompted with the mode, ```train``` being train on dataset and ```chat``` being chat. There is also ```chatreadonly``` for readonly chat (the model weights will not re-save to disk and override things) and ```chatnotrace``` if you want to break things. You will have to configure your own dataset by modifying the code (to run dataset mode), but you should be able to run chat mode without modifying anything if you have weights already.

Once it begins training, you can safely ^C the program and it will save weights. It should also periodically save weights if I'm not mistaken. The saved weights include the internal memory so the model will remember that the next time it runs. You can launch into chat mode and the memory should carry on from whatever it was learning in training.

## How it works

In detail, here are some of the main capabilities of the model that differ from LLMs:
* JEPA-style latent space prediction, as the decoder can be removed/disabled and the model still rolls out forward as is. The model is not trained explicitly on predicting the next byte, but rather on two separate goals (predicting the next 'thing' in latent space, and translating the current latent space vector to a byte).
* Theoretically infinite memory, as it does not have a context window and instead relies on RTUs to store internal state/memory. However it does decay old memories over time. Also I think this should be O(1) memory based on my implementation but I'm not 100% sure.
* Built-in multimodality, as the model outputs bytes (and thus should theoretically be capable of handling any binary data).
* Streaming data live, since the model only processes one byte at once at rapid pace. In fact it is completely 'blind' to everything that came before the current byte, only relying on the current processed byte and its internal memory to decide the next byte. This confirms the model is definitely learning to remember things.
* Continual training, as it keeps training on user input, training data, and its own output to improve its predictions automatically. Keep in mind the model can only output a byte (0-255) each pass anyways (in addition to updating its own state).

The two important hyperparameters are the size of the latent vector (dim) and the amount of individual state layers the latent passes through before decoding (layers). For my 4.5m test these are ```dim = 512``` and ```layers = 16```. There are some other configurations you can change but I think they are less important.

For reproduction purposes the dataset I trained my model on is ```simplewiki-20260801-pages-articles.xml.bz2```, from the Wikipedia dumps.

I think this probably will contribute to solving continual learning and memory but I need other people to review and verify my work! Please feel free to open GitHub issues to tell me what's wrong. If you have compute (e.g. you are a lab or just have GPUs lying around), feel free to fork my code and train larger models as well, with credit. I personally don't have enough compute and as such I can't really train very large models.

Below is an approximate flow chart of the model architecture, made in Apple's Freeform app (excluding the wrapper for dataset cleaning and input/output handling) for reference. Note that the arrow connecting the target latent to the CE loss should instead be the target byte to the CE loss.

<img width="1653" height="1161" alt="JEPA thing" src="https://github.com/user-attachments/assets/2d3a34ff-ba6a-44b8-b361-6c73da9216c0" />


# Experiments:
- The experiment i included are Turning the entire TMT from only using MLX into a Pure numpy for a thorough gradient check on every layer, and Easier Modification.
- I incorporate a New Weight encoding for the layers inside the Model forward pass in hope of achieving a relative max rel error:
- ```python
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
              p[f"layers.{i}.weights.weight"] = self.weight_encoding.weight_shaping(x) # placed Right here.

          self.p = p
    ```
- The Gradient check results (without Weight encoding):
- ```txt
  ok   encoder.embed.weight     max rel err 7.608e-10
  ok   decoder.decode.weight    max rel err 1.730e-07
  ok   decoder.decode.bias      max rel err 4.674e-09
  ok   decoder.stop.weight      max rel err 4.621e-10
  ok   decoder.stop.bias        max rel err 1.053e-11
  ok   layers.0.decay           max rel err 0.000e+00
  ok   layers.0.norm.weight     max rel err 1.930e-09
  ok   layers.0.norm.bias       max rel err 1.687e-09
  ok   layers.0.weights.weight  max rel err 5.710e-09
  ok   layers.1.decay           max rel err 0.000e+00
  ok   layers.1.norm.weight     max rel err 1.296e-08
  ok   layers.1.norm.bias       max rel err 3.820e-09
  ok   layers.1.weights.weight  max rel err 5.372e-08
  ok   layers.2.decay           max rel err 0.000e+00
  ok   layers.2.norm.weight     max rel err 8.695e-08
  ok   layers.2.norm.bias       max rel err 2.099e-08
  ok   layers.2.weights.weight  max rel err 2.966e-09
  
  layer  0 dim    2  num -2.044740e-01  ana -2.044740e-01  rel 1.827e-10
  layer  1 dim   12  num  6.881759e-02  ana  6.881759e-02  rel 2.207e-10
  layer  2 dim   28  num -3.725558e-03  ana -3.725558e-03  rel 4.509e-09
  ```
  - Note:
  - This is NOT the Average results for each Gradient check without weight encoding per N > 1, where N is the total gradient check test.
- The Gradient check results (with Weight encoding):
- ```txt
  ok   encoder.embed.weight     max rel err 8.510e-10
  MEH  decoder.decode.weight    max rel err 1.738e-04
  MEH  decoder.decode.bias      max rel err 1.292e-04
  ok   decoder.stop.weight      max rel err 2.154e-09
  ok   decoder.stop.bias        max rel err 2.555e-09
  ok   layers.0.decay           max rel err 0.000e+00
  ok   layers.0.norm.weight     max rel err 1.803e-09
  ok   layers.0.norm.bias       max rel err 2.434e-10
  ok   layers.0.weights.weight  max rel err 1.157e-08
  ok   layers.1.decay           max rel err 0.000e+00
  ok   layers.1.norm.weight     max rel err 2.612e-08
  ok   layers.1.norm.bias       max rel err 1.508e-10
  ok   layers.1.weights.weight  max rel err 2.934e-07
  ok   layers.2.decay           max rel err 0.000e+00
  ok   layers.2.norm.weight     max rel err 2.755e-09
  ok   layers.2.norm.bias       max rel err 6.421e-10
  ok   layers.2.weights.weight  max rel err 2.163e-07
  
  layer  0 dim   22  num -5.004285e+00  ana -5.004285e+00  rel 8.009e-11
  layer  1 dim   13  num  2.089950e+00  ana  2.089950e+00  rel 4.510e-11
  layer  2 dim   26  num  5.220630e-01  ana  5.220630e-01  rel 2.275e-10 
  ```
  - Note
  - This is the average results where the Model incorporates the weight encoding inside its forward pass per 3 continuous test.

  # Key Results:
  - From the above experimental results, the key results:
    - the decoder.decode.weight has relatively higher max rel error (1.738e-04) when The Model used the Weight encoding.
    - the decoder.decode.bias has relatively higher max rel error (1.292e-04) when the model used the Weight encoding.
  - Conclusion:
    - This does not meant the Gradient when using the Weight encoding is bad, it just happens to have a different quality that was still Within what float32 finite-differencing produces for a correctly-implemented gradient.
