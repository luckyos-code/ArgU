# ArgU

### Pipenv - Basics
* Install Pipenv: ``` $ pip install pipenv ```
* Install dependencies: ``` $ pipenv install ```
* Run command in environment: ``` $ pipenv run <command> ``` (e.g. ``` $ pipenv run python my_project.py ```)
---
* Add package to dependencies: ``` $ pipenv install <package> ```
* Remove package from dependencies: ``` $ pipenv uninstall <package> ```
* Launch Pipenv environment shell: ``` $ pipenv shell ``` (test: ``` $ python --version ```; exit command: ``` $ exit ```)

### Modul excecution
* Main Program: ``` $ python -m argU ```
* For individual moduls, cd into directories and run ``` $ python -m [modulname] ```