# 🎉  1. Introduction
Improve handwriting using GANS

## 🔨 2. Installation process

Install all the dependencies. Also creates the virtual environment if it didn't exist yet.
```
poetry config virtualenvs.in-project true
poetry install
```
## ✅  3. Build and Test

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

## 👥  4. Contribute

Read [here](./CONTRIBUTING.md) how you can contribute to make our code
better.
## ⚗️ 5. Documentation

To see the documentation:

```
poetry run mkdocs serve
```
## 📝  6. References

https://github.com/clovaai/CRAFT-pytorch?tab=readme-ov-file