FROM python:3.7
ARG PYTHON_VERSION=3.7

RUN pip install numpy
RUN pip install pandas
RUN pip install flask
RUN pip install flask-restplus
RUN pip install flask-SSLify
RUN pip install flask-Admin

RUN pip install gunicorn
RUN pip install hdfs
RUN pip install torch==1.3.0
RUN pip install cloudpickle==1.2.2

USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl

COPY . /app
WORKDIR /app

EXPOSE 8051
CMD ["gunicorn", "-b", "0.0.0.0:8051", "app_io"]
