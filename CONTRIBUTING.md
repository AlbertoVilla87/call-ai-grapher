# Contributing
In this project we use the gitflow branching model toghether with forks. If you want a detailed explanation of this methodology, here's a nice article: [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)

## Central repository & Forks
The project has a central repository that serves as the single source of truth. This repo only has these branches:
- `dev`: The most up to date development code. It should always be in a working state.
- `main`: Where the production code lives.

## Commits

Your commit message should follow this format: `[emoji] [action] #[item_id]: phrase`.
> Example: `🐛 Fixes #1111: Removed duplicates`

- `emoji`: Emojis are used to increase the version number automatically based on some predefined rules and following the [Semantic Versioning Specification (SemVer)](https://semver.org/). Adding an emoji is optional, although it's encouraged, especially when making any improvement that could be noticed by the end user. You're free to use an emoji not listed below. In this case the commit message on its own won't trigger a new release.
- `action`: Single-word action, which should always be a verb in the present tense. Some examples: `Fix`, `Closes`, etc.
- `item_id`: ID of the DevOps work item. This will automatically link the commit with the referenced work item. **Do not forget to add the `#` before the ID.**
- `phrase`: All your commit messages should be written in English. Please start the first word of the sentence with a capital letter.

|Increment|Emojis|
|-|-|
|Major|💥 Introduce breaking changes.|
|Minor|✨ Introduce new features.<br/>🏗️ Make architectural changes.<br/>♻️ Refactor code.<br/>⚡️ Improve performance.<br/>👽️ Update code due to external API changes.|
|Patch|🚑️ Critical hotfix.<br/>🔒️ Fix security issues.<br/>🐛 Fix a bug.<br/>🥅 Catch errors.<br/>🔐 Add or update secrets.<br/>📌 Pin dependencies to specific versions.<br/>🔧 Add or update configuration files.<br/>🌐 Internationalization and localization.<br/>💬 Add or update text and literals.<br/>📝Add or update documentation.|

For more emojis, please check out [gitmoji](https://gitmoji.dev/).
To encode and decode the emojis used in the `pyproject.toml` file you could use [DenCode](https://dencode.com/).
