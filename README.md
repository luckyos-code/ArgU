# ArgU

You build a system to support users who directly search for arguments, e.g., by supporting their stance, or by aiding them in building a stance on topics of a general societal interest such as abandonment of plastic bottles, animal experiments, abortion, etc. This subtask targets argumentative conversations. You will retrieve documents from the focused crawl with content from online debate portals (idebate.org, debatepedia.org, debatewise.org) and from Reddit's ChangeMyView. Be sure to retrieve good ''strong'' arguments. Our human assessors will label the retrieved documents manually, both for their general topical relevance, and for argument quality dimensions such as: (1) whether an argumentative text is logically cogent, (2) whether it is rhetorically well-written, and (3) whether it contributes to the users' stance-building process, i.e., somewhat similar to the concept of "utility" (refer to this paper for more information on argument quality 

### Task Formate und Abgabe

<b>qid</b>: The topic number.<br>
<b>Q0</b>: Unused, should always be Q0.<br>
<b>doc</b>: The document id returned by your system for the topic qid:<br>
* For subtask (1): Use the official args-me id.<br>

<b>rank</b>: The rank the document is retrieved at.<br>
<b>score</b>: The score (integer or floating point) that generated the ranking. The score must be in descending (non-increasing) order. It is important to handle tied scores. (trec_eval sorts documents by the score values and not your rank values.)<br>
tag: A tag that identifies your group and the method you used to produce the run.
The fields should be spectated with a whitespace. The width of the columns in the format is not important, but it is important to include all columns and have some amount of white space between the columns.

An example run for task 1 is:
- 1 Q0 10113b57-2019-04-18T17:05:08Z-00001-000 1 17.89 myGroupMyMethod
- 1 Q0 100531be-2019-04-18T19:18:31Z-00000-000 2 16.43 myGroupMyMethod
- 1 Q0 10006689-2019-04-18T18:27:51Z-00000-000 3 16.42 myGroupMyMethod

[Topics Queries XML](resources/topics-automatic-runs-task-1.xml)

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

