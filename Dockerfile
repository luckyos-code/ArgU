### Image to run ArgU
# Start with java image
FROM openjdk:15-jdk-slim

# Install python3.6
COPY --from=python:3.6-slim ./ ./

# Install python packages
WORKDIR /ArgU
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Install needed packages
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	wget \
	unzip

# Install terrier-core in /terrier-core-4.2
WORKDIR /
RUN wget -nv --header "Referer: http://terrier.org/download/agree.shtml?terrier-core-4.2-bin.tar.gz" http://terrier.org/download/files/terrier-core-4.2-bin.tar.gz \
	&& tar -zxvf terrier-core-4.2-bin.tar.gz \
	&& rm terrier-core-4.2-bin.tar.gz

# Nano for testing
RUN apt-get update \
	&& apt-get install nano

# Get args-me.json and topics.xml
WORKDIR /input
RUN wget -nv https://zenodo.org/record/3274636/files/argsme.zip \
	&& unzip argsme.zip \
	&& rm argsme.zip
COPY ./topics.xml /input/topics.xml

# Get the whole app
WORKDIR /ArgU
COPY ./ ./

# Run the necessary steps
CMD bash
#	python argU/preprocessing/mongodb.py -i /input \
#	&& python argU/preprocessing/trec.py \
#	&& python argU/indexing/a2v.py -f \
#	&& /terrier-core-4.2/
#	&& /terrier-core-4.2/
#	&& /terrier-core-4.2/
#	&& /terrier-core-4.2/
#	&& python -m argU -d \
#	&& python -m argU -m -o /output