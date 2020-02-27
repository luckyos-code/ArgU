### Image to run ArgU
# Start with java image
FROM openjdk:15-jdk-slim

# Install python3.6
COPY --from=python:3.6-slim ./ ./

# Install python packages
WORKDIR /ArgU
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Install wget
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	wget

# Install terrier-core in /terrier-core-4.2
WORKDIR /
RUN wget -nv --header "Referer: http://terrier.org/download/agree.shtml?terrier-core-4.2-bin.tar.gz" http://terrier.org/download/files/terrier-core-4.2-bin.tar.gz \
	&& tar -zxvf terrier-core-4.2-bin.tar.gz \
	&& rm terrier-core-4.2-bin.tar.gz

# Download NLTK data
RUN python -m nltk.downloader 'averaged_perceptron_tagger' 'wordnet'

# Get the whole app
WORKDIR /ArgU
COPY ./ ./

# Run the necessary steps
CMD python argU/preprocessing/mongodb.py -i /input \
	&& python argU/preprocessing/trec.py -i /input \
	&& python argU/indexing/a2v.py -f \
	&& /terrier-core-4.2/bin/trec_setup.sh /ArgU/resources/args-me.trec \
	&& /terrier-core-4.2/bin/trec_terrier.sh -i \
	&& /terrier-core-4.2/bin/trec_terrier.sh -r -Dtrec.model=DPH -Dtrec.topics=/ArgU/resources/topics.trec \
	&& cp /terrier-core-4.2/var/results/DPH_0.res /ArgU/resources/terrier.res \
	&& rm /terrier-core-4.2/var/results/DPH_0.res \
	&& python -m argU -d \
	&& python -m argU -m -s $RUNTYPE -o /output