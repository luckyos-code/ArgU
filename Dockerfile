# Image to run python
FROM python:3.6

WORKDIR /ArgU
COPY ./ ./

RUN pip install -r requirements.txt

CMD ["bash"]