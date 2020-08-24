### Image to run ArgU
# Get python3.6
FROM python:3.6-slim

# Install python packages
WORKDIR /ArgU
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Install unzip
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	unzip

# Get the whole app
WORKDIR /ArgU
COPY ./ ./

# Unpack resources
RUN unzip resources/mapping.zip -d resources \
	&& unzip resources/terrier.zip -d resources \
	&& unzip resources/desm.zip -d resources

# Run the app
CMD python main.py eval --args_topn=$TOP_N --emb=$EMBEDDING --sent=$RUN_TYPE --out=/output