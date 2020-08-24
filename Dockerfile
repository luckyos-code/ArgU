### Image to run ArgU
# Get python3.6
FROM from=python:3.6-slim

# Install python packages
WORKDIR /ArgU
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Get the whole app
WORKDIR /ArgU
COPY ./ ./

# Unpack resources
RUN unzip resources/mapping.zip \
	&& unzip resources/terrier.zip \
	&& unzip resources/desm_results_in.zip \
	&& unzip resources/desm_results_out.zip

# Run the app
CMD python main.py eval --emb=$EMBEDDING --sent=$RUN_TYPE --out=/output