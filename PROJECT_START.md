Install pre-commit:

1. run brew install pre-commit
2. run pre-commit install

Search for `python-template` and change it everywhere to your `project-name`.

Under `src` change the name of your package `<packagename>`.

On GitHub switch on:
- On main setting
  - Delete Head Branch after merging
  - Untick merge commits
- In Branches > protect main
  - Require PR
  - Require Linear History
  - Do not allow bypassing of the rules