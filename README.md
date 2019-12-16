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

### Relevant Files
* [arguments](https://zenodo.org/record/3274636#.XeAyUi03v4a)
* [queries](https://github.com/webis-de/SIGIR-19/blob/master/Data/topics.csv)
* [training data](https://git.informatik.uni-leipzig.de/lg80beba/argument-quality-evaluation/tree/master/Dataset%20Final%20Study)
* [args frontend](https://git.webis.de/args)