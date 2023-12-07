# Introduction
Improve handwriting using GANS
## Project Structure

The main project folder contains the following files and folders:

- **call_ai_grapher/**: All your code goes here. You are free to create packages inside of this directory, for example for data preparation, modeling, or utils. Make sure you create an empty `__init__.py` file inside every package.
- **fakes/**: store fakes images.
- **handwriting/**: store good handwriting images.
- **myhandw/**: store personal handwriting images.
- **gif/**: store gifs files to visualize experiments.
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

![Experiment 1](./gif/evol.gif)

## Run application

To train a neural network.
```
poetry run train
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
