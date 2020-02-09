# Image to run ArgU
FROM python:3.6

# Install requirements
WORKDIR /ArgU
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Download argsme corpus
WORKDIR /ArgU/resources
RUN wget https://zenodo.org/record/3274636/files/argsme.zip \
	&& unzip argsme.zip \
	&& rm argsme.zip

# Nano for testing
RUN apt-get update \
	&& apt-get install nano

# Convert input to csv
COPY ./argU/preprocessing/args_to_csv.py ./argU/preprocessing/args_to_csv.py
RUN python argU/preprocessing/args_to_csv.py \
	&& rm args-me.json

# Get the whole app
WORKDIR /ArgU
COPY ./ ./

# Run the necessary steps
# Keeps container running as bash
CMD python -m argU index -c=all \
	&& python -m argU retrieve -n -1 -a 0.1 \
	&& bash
### Docker

1. ` $ docker build -t argu . `
	- Build the image
	- Only needed once or after changes on host
2. ` $ docker run --name argu -d --rm -it argu `
	- Starts the container as a form of background service
	- Keeps running in the background until stopped
3. ` $ docker exec -it argu bash `
	- Access container shell (shell exit command: '` $ exit `)
	- 'Do your stuff' mode
	- While the container keeps running:
		- enter shell as often as you want
		- changes stay
4. ` $ docker stop argu `
	- Stop the container (removes itself)