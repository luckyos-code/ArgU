# Image to run ArgU
FROM python:3.6

WORKDIR /ArgU
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

WORKDIR /ArgU/resources
RUN wget https://zenodo.org/record/3274636/files/argsme.zip \
	&& unzip argsme.zip \
	&& rm argsme.zip

RUN apt-get update \
	&& apt-get install nano

WORKDIR /ArgU
COPY ./ ./

CMD ["bash"]