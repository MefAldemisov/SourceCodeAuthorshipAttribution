FROM tensorflow/tensorflow:latest-gpu

RUN mkdir -p /usr/app
WORKDIR /usr/app/src

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

CMD [ "python3", "/usr/app/src/main.py"]
