# Image to run ArgU
FROM python:3.6

# Install requirements
WORKDIR /ArgU
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Nano for testing
RUN apt-get update \
	&& apt-get install nano

# Download argsme corpus
WORKDIR /ArgU/resources
RUN wget https://zenodo.org/record/3274636/files/argsme.zip \
	&& unzip argsme.zip \
	&& rm argsme.zip

# Convert input to csv
WORKDIR /ArgU
COPY ./argU/preprocessing/args_to_csv.py ./argU/preprocessing/args_to_csv.py
RUN python argU/preprocessing/args_to_csv.py 

# Get the whole app
WORKDIR /ArgU
COPY ./ ./

# Run the necessary steps
# Keeps container running as bash
CMD python -m argU index -c=all \
	&& python -m argU retrieve -n -1 -a 0.1 \
	&& bash
