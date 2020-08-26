# SentArg: A Hybrid Doc2Vec/DPH Model with Sentiment Analysis Refinement

**Argument retrieval model for [Touché @ CLEF 2020](https://touche.webis.de/) - 1st Shared Task on Argument Retrieval.**

* [Touché notebook](reports/staudte_lange_sentarg.pdf) (English, short paper)

* [Evaluation results](evaluation/evaluation_results.md) - Official results as team 'Oscar François de Jarjayes' in [leader board](https://events.webis.de/touche-20/shared-task-1.html#results)
* [Tira](evaluation/tira_evaluation.md) - Tira evaluation results (on original dataset)
* [Experiments](evaluation/experiments/) - Conducted experiments for evaluation
* [Old lab report](reports/lab_report.pdf) (German)

## Prerequisites

1. Download and extract the [Args.me Corpus](https://zenodo.org/record/3274636/files/argsme.zip)
2. Put args-me.json and the topic.xml in your input directory

## Tira Runs
1. ` $ docker pull mongo:4.4 `
2. ` $ docker build -t argu . `
3. ` $ chmod +x ./tira_run.sh `
4. ` $ ./tira_run.sh -s <run-type> -i $inputDataset -o $outputDir `
	- Run Types
		- ` no `: No sentiments
		- ` emotional `: Emotional is better
		- ` neutral `: Neutral is better

## Docker

1. ` $ docker build -t argu . `
	- Build the image
2. ` $ docker run --name argu-mongo -p 27017:27017 -d --rm mongo `
	- Starts a MonoDB container
3. ` $ docker run -e RUN_TYPE=<run-type> -v <input-dir-path>:/input -v <output-dir-path>:/output --name argu --rm -it --network="host" argu `
	- Runs the ArgU container
	- Run Types
		- ` no `: No sentiments
		- ` emotional `: Emotional is better
		- ` neutral `: Neutral is better
	- Input directory with args-me.json and topics.xml
	- Output directory will get the results as run.txt
4. ` $ docker stop argu-mongo `
	- Remove MongoDB container

## Command Line

0. ` $ pip install -r requirements.txt `

1. ` $ python argU/preprocessing/mongodb.py -i <input-dir-path> `
	- Create mapping (mongoDB ID <--> argument.id); store into MongoDB
	- Store arguments with the new ID into MongoDB
	- Clean arguments and store as train-arguments into MongoDB
	- Read Sentiments and store into MongoDB
2. ` $ python argU/preprocessing/trec.py -i <input-dir-path> `
	- Create a .trec-file for Terrier (train and queries)
3. ` $ python argU/indexing/a2v.py -f `
	- Generate CBOW
	- Generate argument embeddings and store them into MongoDB
4. Install and run Terrier (see [Dockerfile](Dockerfile))
	- Calculate DPH for queries
	- copy result file in [resources](resources/)
5. ` $ python -m argU -d `
	- Compare given queries with argument embeddings; store Top-N DESM scores into MongoDB
6. Merge DESM, Terrier and Sentiments to create final scores
	1. ` $ python -m argU -m -s no -o <output-dir-path> `
		* R1: No sentiments
	2. ` $ python -m argU -m -s emotional -o <output-dir-path> `
		* R2: Emotional is better
	3. ` $ python -m argU -m -s neutral -o <output-dir-path> `
		* R3: Neutral is better
  
### Submodul Excecution

* For individual moduls, cd into directory and run ` $ python -m [modulname] `

## Documentation

### Modules

* [indexing](argU/indexing/) - Index for DESM scores
* [preprocessing](argU/preprocessing/) - Prework and cleaning of input data
* [sentiment](argU/sentiment/) - Sentiment analysis
* [utils](argU/utils/) - Helper functionalities
* [visualization](argU/visualization/) - Visualization of scores

### Built With

* [Docker](https://www.docker.com/) - Used to build and run
* [gensim](https://radimrehurek.com/gensim/) - Used to generate a Continuous Bag of Words Model 
* [NumPy](https://numpy.org) - Used as a mathematical base to compute vectors and matrices
* [Matplotlib](https://matplotlib.org) - Used to visualize scores
* [Google Cloud Natural Language API](https://cloud.google.com/natural-language/) - Used for sentiment analysis
* [Natural Language Toolkit](https://www.nltk.org) - (Deprecated) Used to train sentiment analysis model
* [MongoDB](https://www.mongodb.com) - Used to store arguments and scores
* [Terrier](http://terrier.org) - Used to calculate DPH scores

## Relevant Information for Subtask (1)

### Task

Decision making processes, be it at the societal or at the personal level, eventually come to a point where one side will challenge the other with a why-question, which is a prompt to justify one’s stance. Thus, technologies for argument mining and argumentation processing are maturing at a rapid pace, giving rise for the first time to argument retrieval. We invite to participate in the first lab on **Argument Retrieval** at CLEF 2020 featuring two subtasks:

(1) retrieval in a focused argument collection to support argumentative conversations.

The **(1) subtask** is motivated by the support of users who search for arguments directly, e.g., by supporting their stance, and targets argumentative conversations. The task is to retrieve arguments from the provided dataset of the focused crawl with content from online debate portals for the 50 given topics, covering a wide range of controversial issues. 

### Data

Argument topics for subtask (1) and comparative questions for subtask (2) will be send to each team via email upon completed registration. The topics will be provided as XML files.

Example topic for **subtask (1)**:

   <topic>
      <num>1</num>
      <title>Is climate change real?</title>
      <description>You read an opinion piece on how climate change is a hoax and disagree. Now you are looking for arguments supporting the claim that climate change is in fact real.</description>
      <narrative>Relevant arguments will support the given stance that climate change is real or attack a hoax side's argument.</narrative>
   </topic>

**Document collections.** To search for relevant arguments, you can use your own index based on the dataset args-me or for simplicity deploy an API of the search engine args.me.

### Runs Submission

We encourage participants to use TIRA for their submissions to increase replicability of the experiments. We provide a dedicated TIRA tutorial for Touché and are available to walk you through. You can also submit runs per email. In both cases, we will review your submission promptly and provide feedback.

Runs may be either automatic or manual. An automatic run is made without any manual manipulation of the given topic titles. Your run is automatic if you do not use description and narrative for developing approaches. A manual run is anything that is not an automatic run. Please let us know which of your runs are manual upon submission.

The submission format for both tasks will follow the standard TREC format:

`qid Q0 doc rank score tag`

With:

* qid: The topic number.
* Q0: Unused, should always be Q0.
* doc: The document id returned by your system for the topic qid:
	* For **subtask (1)**: Use the official args-me id.
* rank: The rank the document is retrieved at.
* score: The score (integer or floating point) that generated the ranking. The score must be in descending (non-increasing) order. It is important to handle tied scores. (trec_eval sorts documents by the score values and not your rank values.)
* tag: A tag that identifies your group and the method you used to produce the run.

The fields should be spectated with a whitespace. The width of the columns in the format is not important, but it is important to include all columns and have some amount of white space between the columns.

An example run for task 1 is:
```
1 Q0 10113b57-2019-04-18T17:05:08Z-00001-000 1 17.89 myGroupMyMethod
1 Q0 100531be-2019-04-18T19:18:31Z-00000-000 2 16.43 myGroupMyMethod
1 Q0 10006689-2019-04-18T18:27:51Z-00000-000 3 16.42 myGroupMyMethod
...
```

### Material
* Literature: [Ajjour et al. 2019](https://webis.de/downloads/publications/papers/stein_2019o.pdf), [Wachsmuth et al. 2017](https://webis.de/downloads/publications/papers/stein_2017r.pdf), [Potthast et al. 2019](https://webis.de/downloads/publications/papers/stein_2019j.pdf)
* Dataset: [Args.me Corpus](https://zenodo.org/record/3274636#.XeAyUi03v4a)
* Evaluation: [Topics Queries XML](/topics.xml)
