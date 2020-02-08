# Image to run python
FROM python:3.6

WORKDIR /ArgU
COPY ./ ./

WORKDIR /ArgU/resources
RUN wget https://zenodo.org/record/3274636/files/argsme.zip \
	&& unzip argsme.zip \
	&& rm argsme.zip

WORKDIR /ArgU
RUN pip install -r requirements.txt

CMD ["bash"]