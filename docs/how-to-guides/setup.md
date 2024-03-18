# Project Setup

!!! info "About This Section"
    Setup a DocIQ Use Case based on [DocIQ Backend Template](https://dev.azure.com/edaa-eudl-devops/ai-dsmle-dociq/_git/ai-dsmle-dociq-template`).

## Steps

!!! warning
      If you select <install_project=No>, follow steps 1-4. If you <install_project=Yes> go to steps 2 and 4.

## 1. Installation dependencies

Install all the dependencies. Also creates the virtual environment if it didn't exist yet.

```
poetry config virtualenvs.in-project true
poetry install
```

!!! warning
    _If the installation fails you are probably missing the required Python version. Find the required version by running `pyenv version`, and then install it by running `pyenv install x.y.z`, where x.y.z should be replaced with the version number. Depending on your internet connection and your machine the installation can take a few minutes._

## 2. Install the pre-commit hooks

```
poetry run pre-commit install
```

## 3. Setting API secrets

!!! info "About This Section"
    Please, go to the following link [How to recreate access to GenAI Lounge](https://rmp-confluence.zurich.com/pages/viewpage.action?spaceKey=LLMPLT&title=Onboarding+Use+Cases+to+the+Dev+Environment)

After setup repo:

- add cleaning script to git config (if not already added)

```
chmod +x tools/hide_secrets_before_push.sh
git config filter.hide_secrets.clean tools/hide_secrets_before_push.sh
```

- fill the .local.env file with secrets, copy outside of repository and symlink it to the repository

```
ln -s ../.local.env ./information_extrator_module/.local.env
```

## 4. Build and Test

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
  
## 5. Create a remote Azure repository from a local folder

### 5.1 Navigate to Repositories

   Once your project is set up, navigate to the "Repos" section:

- Click on the "New Repository" button.
- Fill in the required details such as Repository name, Description, and Version control (select Git).

### 5.2 Initialize Git Repository Locally

   If your local folder is not already a Git repository, you need to initialize it:

   ```bash
   cd /path/to/local/folder
   git init
   ```

### 5.3 Add Azure DevOps Repository as Remote

- On the Azure DevOps website, go to your new repository and copy the repository URL (should be something like `https://dev.azure.com/yourorganization/yourproject/_git/yourrepository`).
- In your local Git repository folder, add the Azure DevOps repository as a remote:

     ```bash
     git remote add origin https://dev.azure.com/yourorganization/yourproject/_git/yourrepository
     ```

### 5.4 Push Local Repository to Azure DevOps

- Push your local repository to the Azure DevOps repository:

     ```bash
     git push -u origin --all
     git push -u origin --tags
     ```

   This will push your local repository to the Azure DevOps remote repository.
