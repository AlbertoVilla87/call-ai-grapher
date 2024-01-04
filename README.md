# Introduction
Improve handwriting using GANS
## Project Structure

The main project folder contains the following files and folders:

- **call_ai_grapher/**: All your code goes here. You are free to create packages inside of this directory, for example for data preparation, modeling, or utils. Make sure you create an empty `__init__.py` file inside every package.
- **tests/**: In this directory you will write your unit tests. As a best practices, make sure to reflect the same structure as in the `call_ai_grapher/` directory. Prefix module, class and function names with "test" following the.
- **.gitignore**: Indicates which files and folders shouldn't be added to the Git repository.
- **.pre-commit-config.yaml**: Used to configure which checks and actions should run before every Git commit.
- **.python-version**: File where `pyenv` stores the Python version used by the project.
- **build-pipeline.yml**: Azure DevOps Pipeline for building the project.
- **CHANGELOG.md**: Automatic generated file when a new build has been created.
- **poetry.lock**: File used by `poetry` to fix the versions of the packages used in this project.
- **pyproject.toml**: Contains all the configuration for the entire project. It replaces `setup.py`, `requirements.txt`, and package-specific configuration files.


## Installation process

Install all the dependencies. Also creates the virtual environment if it didn't exist yet.
```
poetry install
```

_If the installation fails you are probably missing the required Python version. Find the required version by running `pyenv version`, and then install it by running `pyenv install x.y.z`, where x.y.z should be replaced with the version number. Depending on your internet connection and your machine the installation can take a few minutes._

Install the pre-commit hooks.
```
poetry run pre-commit install
```

## Experiments

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

![Experiment 4](./gif/exp_4_losses.png) 

### Experiment 5

build a Wasserstein GAN with Gradient Penalty (WGAN-GP) (https://arxiv.org/abs/1701.07875) that solves the vanishing gradient issue with the GANs seen in experiment 4.

![Experiment 5](./gif/exp_5_losses.png)

We can see as the discriminator is able to reduce the losses when picture is changing, providing feedback to generator to adapt the new style. However, we continue to see a lot of noise which could be removed adding to the generator a denoising autoencoder module https://plainenglish.io/blog/denoising-autoencoder-in-pytorch-on-mnist-dataset-a76b8824e57e. We need to analyze why in step 490 the loss discriminator increase and then is constant.

| Experiment | Description | Results | 
| -------- | -------- | -------- |
|  1   | GANS with two discriminators | ![Experiment 1](./gif/evol.gif)   |
|  2   | GANS with convolution and two discriminators |![Experiment 2](./gif/exp_2.gif)   |
|  3   | GANS with convolution and one discriminator |![Experiment 3](./gif/exp_3.gif)   |
|  4   | GANS with one discriminator |![Experiment 4](./gif/exp_4.gif)   |
|  5   | GANS with WGAN-GP |![Experiment 5](./gif/exp_5.gif)   |


### Experiment 6

Include Autoencoder Denosing.

![Experiment 6](./gif/exp_6_losses.png)

![Experiment 6](./gif/exp_6.gif)

We can observe a high noise removal performance within a few epochs of training. However, the letters 'c' and 'e' are very similar. This might be due to the limited variability of the sample, as only one sample per character is available.

## Run application

To train a neural network.
```
poetry run train
```

## Run Jupyter

```
poetry run jupyter notebook
```
## Software dependencies
- Install [Poetry](https://python-poetry.org/docs/#installation).
## Resources
- [Poetry - Basic usage](https://python-poetry.org/docs/basic-usage/)
- [pyenv - Usage](https://github.com/pyenv/pyenv#usage)

# Build and Test

Please activate the virtual environment by using `poetry shell` before running the commands below, or prefix all commands with `poetry run`.

- Run pre-commit hooks for all files (not only staged files) manually.
  ```
  pre-commit run --all-files
  ```
- Run all unit tests.
  ```
  pytest
  ```
- Measure code coverage.
  ```
  coverage run -m pytest
  ```
- Visualize code coverage.

  - View code coverage summary in terminal.
    ```
    coverage report
    ```
  - Generate HTML code coverage report.
    ```
    coverage html
    ```
  - View code coverage directly inside your code.
    ```
    coverage xml
    ```
    _Install the Coverage Gutters extension if you are using Visual Studio Code, and click on "Watch" on the left side of the status bar in the bottom of the IDE to visualize the code coverage._

# Contribute

Read [here](./CONTRIBUTING.md) how you can contribute to make our code better.