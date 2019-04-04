## Handwriting generation using LSTM

Implementing [this paper](https://arxiv.org/pdf/1308.0850.pdf) by Alex Graves using pytorch.

### Generating samples

Use `generate.py` for generating from models using saved state_dict files.

```bash
usage: generate.py [-h] [--uncond] --model_path MODEL_PATH
                   [--text TEXT [TEXT ...]] [--sample_length SAMPLE_LENGTH]
                   [--num_sample NUM_SAMPLE] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --uncond              If want to generate using the unconditional model.
                        Default is conditional
  --model_path MODEL_PATH
                        path to the saved sate_dict file to be used for
                        generating samples
  --text TEXT [TEXT ...]
                        text for which handwriting to be synthesized (for
                        conditional model)
  --sample_length SAMPLE_LENGTH
                        sample length for unconditional model
  --num_sample NUM_SAMPLE
                        number of samples to generate from unconditional model
  --seed SEED
```

Example command for generating from the saved best conditional model:

```bash
python generate.py --model_path data/model_files/handwriting_cond_best.pt --text "Hello world" "RNNs are awesome!"
```

Example command for generating from the saved best unconditional model:

```bash
python generate.py --uncond --model_path data/model_files/handwriting_uncond_best.pt --sample_length 600 --num_sample 4
```