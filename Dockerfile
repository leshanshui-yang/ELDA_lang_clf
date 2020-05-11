FROM python:3.7
ARG PYTHON_VERSION=3.7

RUN pip install numpy
RUN pip install pandas
RUN pip install Flask
RUN pip install flask-restplus
RUN pip install Flask-SSLify
RUN pip install Flask-Admin
RUN pip install gunicorn
RUN pip install hdfs

RUN pip install torch==1.3.0
RUN pip install git+https://github.com/NVIDIA/apex
RUN pip install cloudpickle==1.2.2

RUN pip install transformers

USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl

COPY . /app
WORKDIR /app

EXPOSE 2333
CMD ["gunicorn", "-b", "0.0.0.0:2333", "pred"]
