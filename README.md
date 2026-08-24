# Call AI Grapher

Improve handwriting using GANS.

## Roadmap

Goal: an application that takes a scanned document written in poor handwriting and improves it towards a pretty handwriting style, with a regulator (`alpha`) that lets the user choose how much improvement is applied.

Pipeline: `scanned page -> character detection -> character classification -> per-character stylization (alpha) -> page recomposition -> Gradio app`

| Stage | Status | Branch |
| ----- | ------ | ------ |
| Pipeline skeleton (types, stages, orchestrator, CLI) | ✅ | `ft/pipeline-skeleton` |
| Spanish alphabet dataset builder (A-Z + ñ, TTF + real samples) | ✅ | `ft/alphabet-dataset` |
| Character detector (YOLOv8, MSER fallback) | ✅ | `ft/char-detector` |
| Character classifier (CNN) | ✅ | `ft/char-classifier` |
| Per-character style transfer (pix2pix / latent AE) | ✅ | `ft/style-stylizer` |
| Blend regulator (latent interpolation alpha) | ✅ | `ft/blend-regulator` |
| Document recomposer (baseline alignment) | ✅ | `ft/document-recomposer` |
| Gradio app (upload, slider, before/after) | ✅ | `ft/web-ui` |

# Getting started

## 1. Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then install all the dependencies. This also creates the virtual environment (`.venv`) if it didn't exist yet:

```
uv sync --all-groups
```

_If the installation fails you are probably missing the required Python version. uv installs it automatically. Depending on your internet connection and your machine the installation can take a few minutes._

Install the pre-commit hooks:

```
uv run pre-commit install
```

## 2. Quick start (no training needed)

The easiest way in is the local web app. Upload a scanned page and use the improvement slider to compare before/after:

```
uv run ui
```

Then open the URL printed in the terminal (http://127.0.0.1:7860 by default).

Prefer the command line? Run the pipeline directly (MSER detection + baseline cleanup stylizer out of the box):

```
uv run improve-document --input documents/page.jpeg --output documents/improved.png --alpha 0.8
```

Everything below is optional: each step trains one model that makes the result better, and every step tells you exactly how to plug it into the command above.

# Training your own models

Follow these steps in order; each one builds on the previous and ends with the `improve-document` flags that activate it.

## Step 1. Alphabet dataset

Build the alphabet glyphs every other model trains on. Drop your pretty style `.ttf` fonts into `fonts/` first:

```
uv run build-alphabet --fonts "fonts/**/*.ttf" --out dataset/alphabet --size 64
```

Ingest your own handwriting crops later with `--samples samples/`, where `samples/` contains one directory per character class.

## Step 2. Character detection (YOLOv8)

The detector locates characters on a scanned page (single class: `character`);
labeling each one is the classifier's job (next step). Training data is
generated synthetically from the alphabet glyphs, so no manual annotation is
needed.

1. Generate synthetic training pages from the alphabet glyphs (scan-like degradation included):

   ```
   uv run generate-detector-dataset --glyphs dataset/alphabet --out dataset/detector --train-pages 500 --val-pages 100
   ```

2. Train the detector. Reference run on CPU: 5 epochs over 80 pages took ~1 min
   and already reached `mAP50 = 0.78`, `recall = 0.96`; expect near-perfect
   boxes with the settings below (~30-60 min on a modern laptop CPU):

   ```
   uv run train-detector --data dataset/detector/data.yaml --epochs 100 --imgsz 480 --out models/character_detector.pt
   ```

   Best weights are copied to `models/character_detector.pt`. Training curves
   land in `runs/detect/` (open with `tensorboard --logdir runs`).

3. Run the pipeline with the trained detector (`--detector mser` is the
   default fallback when no model is trained yet):

   ```
   uv run improve-document --input documents/page.jpeg --output documents/improved.png \
       --alpha 0.8 --detector yolo --model models/character_detector.pt --confidence 0.25
   ```

## Step 3. Character classification (A-Z + Ñ)

A small CNN (`CharCNN`, ~300k parameters) labels every detected character
crop using the alphabet dataset from `build-alphabet`. It reads the same
normalized images, so fonts and your real handwriting samples are both used
automatically.

1. Train on the alphabet dataset (seconds on CPU; reference: 83% accuracy
   over 27 classes with only 2 fonts — add more fonts and real samples in
   `--samples` to improve it):

   ```
   uv run train-classifier --data dataset/alphabet --out models/char_classifier.pt --epochs 40
   ```

2. Pass the checkpoint to the pipeline to label characters:

   ```
   uv run improve-document --input documents/page.jpeg --output documents/improved.png \
       --alpha 0.8 --detector yolo --classifier models/char_classifier.pt
   ```

## Step 4. Style transfer (ugly -> pretty, pix2pix)

A conditional GAN (`UNetGenerator` + `PatchDiscriminator`, pix2pix recipe)
learns to translate your poor handwriting towards the pretty style. One
shared network handles every character: the input crop carries the letter
shape and the network only learns the style mapping.

Training pairs are synthesized automatically: each clean alphabet glyph is
degraded with handwriting-like distortions (elastic deformation, slant,
thickness changes, scan noise) to produce its "ugly" counterpart, so both
sides of every pair are pixel-aligned without manual annotation.

1. Train on the alphabet dataset (~2 min CPU for 15 epochs; L1 drops from
   0.40 to 0.03 meaning the generator reproduces the pretty style):

   ```
   uv run train-stylizer --data dataset/alphabet --out models/char_stylizer.pt --epochs 30
   ```

2. Run the pipeline with neural stylization. The regulator `--alpha` becomes
   a cross-fade between your original stroke (0) and the full pretty style
   (1):

   ```
   uv run improve-document --input documents/page.jpeg --output documents/improved.png \
       --alpha 0.7 --detector yolo --classifier models/char_classifier.pt \
       --stylizer neural --stylizer-model models/char_stylizer.pt
   ```

   With `--stylizer baseline` (default) the regulator blends against a simple
   binarized cleanup instead, useful when no model is trained yet.

## Step 5. Latent blend regulator (no ghosting at mid alphas)

The pixel cross-fade above doubles the exposure at mid alphas (both strokes
show through). The latent backend fixes this: a convolutional autoencoder —
trained to reconstruct both degraded and clean glyphs — encodes your ugly
crop and the pretty reference glyph of its class, alpha interpolates between
the two encodings, and the decoder renders a coherent intermediate
handwriting. The reference glyph per character is looked up in the alphabet
dataset using the classifier label.

1. Train the autoencoder (~30 s CPU for 20 epochs; reconstruction L1 drops
   from 0.37 to 0.04):

   ```
   uv run train-autoencoder --data dataset/alphabet --out models/char_autoencoder.pt --epochs 30
   ```

2. Run the pipeline with latent stylization. This backend requires the
   classifier so each character finds its reference glyph:

   ```
   uv run improve-document --input documents/page.jpeg --output documents/improved.png \
       --alpha 0.5 --detector yolo --classifier models/char_classifier.pt \
       --stylizer latent --ae-model models/char_autoencoder.pt --alphabet-dir dataset/alphabet
   ```

The web app mirrors all of this: its "Backends" panel exposes the same options
as the CLI flags (detector `mser`/`yolo` with weights path and confidence,
optional classifier checkpoint, and stylizer `baseline`/`neural`/`latent` with
its model paths and the alphabet dataset).

# Project structure

The repository is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) split into backend and frontend:

```bash
.
├── pyproject.toml            # workspace root + shared tooling config
├── assets/                   # gifs, handwriting samples and base weights
├── backend/
│   ├── pyproject.toml        # call-ai-grapher package + CLI entry points
│   ├── experiments/          # early GAN/denoising experiment scripts
│   └── src/call_ai_grapher/  # pipeline, models, dataset, object detection...
├── frontend/
│   ├── pyproject.toml        # call-aigrapher-ui package (Gradio)
│   └── src/call_aigrapher_ui/
└── deploy/databricks/        # requirements.txt generated by pre-commit
```

Generated artifacts (`dataset/`, `models/`, `runs/`, `junit/`) stay at the repository root and are git-ignored.

# Build and Test

All commands run through `uv run`, no manual activation needed.

- Check what the linters would change across all files (not only staged files).
  ```
  uv run pre-commit run --all-files
  ```
- Run all unit tests.
  ```
  uv run pytest
  ```
- Measure code coverage.
  ```
  uv run coverage run -m pytest
  ```
- Visualize code coverage.

  - View code coverage summary in terminal.
    ```
    uv run coverage report
    ```
  - Generate HTML code coverage report.
    ```
    uv run coverage html
    ```
  - View code coverage directly inside your code.
    ```
    uv run coverage xml
    ```
    _Install the Coverage Gutters extension if you are using Visual Studio Code, and click on "Watch" on the left side of the status bar in the bottom of the IDE to visualize the code coverage._

# Experiments

The research history behind the product, oldest first. Each script lives in `backend/experiments/`.

### Experiment 1

My first experiment consists on improving my "a" handwriting. The first step is to teach generator to create my "a" and then improving it with a better style. In the following gif we can see how the generator learns (upper graph) based on "a" reference used by discriminator (lower graph). A metaphor for how AI and humans can go hand in hand.

### Experiment 2

Second experiment consists on a Deep Convolutional GAN (DCGAN). Main features:

• Replace any pooling layers with strided convolutions (discriminator) and fractional-strided
convolutions (generator).<br>
• Use BatchNorm in both the generator and the discriminator.<br>
• Remove fully connected hidden layers for deeper architectures.<br>
• Use ReLU activation in generator for all layers except for the output, which uses Tanh.<br>
• Use LeakyReLU activation in the discriminator for all layers.<br>

DCGAN uses convolutions which do not depend on the number of pixels on an image. However, the number of channels is important to determine the size of the filters.

We can see a checkerboard when the image passes from poor handwriting to the pretty style one. We could not initialize the discriminator to avoid this.

### Experiment 3

Same GANS model without creating a new Discriminator instance when we change the style. We continue to see a very abrupt jump.

### Experiment 4

We go back to GANS of experiment 1. However, in this case, we have a vanishing gradient issue. When we change the image, The discriminator is unable to distinguish that change and is fooled by the generator. To avoid this, we can apply Wasserstein GAN with Gradient Penalty.

![Experiment 4](./assets/gif/exp_4_losses.png)

### Experiment 5

Build a Wasserstein GAN with Gradient Penalty (WGAN-GP) (https://arxiv.org/abs/1701.07875, https://arxiv.org/pdf/1704.00028.pdf, https://lilianweng.github.io/posts/2017-08-20-gan/) that solves the vanishing gradient issue with the GANs seen in experiment 4.

![Experiment 5](./assets/gif/exp_5_losses.png)

We can see as the discriminator is able to reduce the losses when picture is changing, providing feedback to generator to adapt the new style. However, we continue to see a lot of noise which could be removed adding to the generator a denoising autoencoder module https://plainenglish.io/blog/denoising-autoencoder-in-pytorch-on-mnist-dataset-a76b8824e57e. We need to analyze why in step 490 the loss discriminator increase and then is constant.

| Experiment | Description | Results |
| -------- | -------- | -------- |
|  1   | GANS with two discriminators | ![Experiment 1](./assets/gif/evol.gif)   |
|  2   | GANS with convolution and two discriminators |![Experiment 2](./assets/gif/exp_2.gif)   |
|  3   | GANS with convolution and one discriminator |![Experiment 3](./assets/gif/exp_3.gif)   |
|  4   | GANS with one discriminator |![Experiment 4](./assets/gif/exp_4.gif)   |
|  5   | GANS with WGAN-GP |![Experiment 5](./assets/gif/exp_5.gif)   |

### Experiment 6

Include Autoencoder Denoising.

![Experiment 6](./assets/gif/exp_6_losses.png)

![Experiment 6](./assets/gif/exp_6.gif)

We can observe a high noise removal performance within a few epochs of training. However, the letters 'c' and 'e' are very similar. This might be due to the limited variability of the sample, as only one sample per character is available.

Reproduce it today with:

```
uv run python backend/experiments/train.py     # GANS training (experiments 1-5)
uv run python backend/experiments/denoise.py   # autoencoder denoising (experiment 6)
```

### Experiment 7

We research about object detection. We discover an important part before detection: region proposals. **Region proposal** is a technique that helps in identifying islands of regions where the pixels are similar to one another. *SelectSearch* is a region proposal algorithm used for object localization where it generates proposals of regions that are likely to be grouped together based on their pixel intensitites. However, our case is simpler and we can apply the following technique:

https://stackoverflow.com/questions/40443988/python-opencv-ocr-image-segmentation


<img src="./assets/gif/exp_7.jpeg" alt="Experiment 7" width="500" />

We can see a high level character recognition but we still seeing areas with multiple characters. Therefore, Object Detection with RNN is needed.

## Run Jupyter (FYI)

```
uv run jupyter notebook
```

# Resources

- [uv - Basic usage](https://docs.astral.sh/uv/guides/projects/)
- [pyenv - Usage](https://github.com/pyenv/pyenv#usage)
- [CRAFT text detector](https://github.com/clovaai/CRAFT-pytorch?tab=readme-ov-file)

# Contribute

Read [here](./CONTRIBUTING.md) how you can contribute to make our code better.
