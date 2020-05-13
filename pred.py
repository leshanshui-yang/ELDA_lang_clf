import os
import pandas as pd
import numpy as np  
from flask import Flask, jsonify, request
from flask import current_app, abort, Response
import cloudpickle
import warnings
import logging
import subprocess
import transformers
from hdfs import InsecureClient
import importlib
import requests


logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
application = Flask(__name__)
# hdfs_uri = 'http://nn1:50070'

@application.route('/config', methods=['POST'])
def config():
    # Read params from Json
    if not request.json or not 'preprocess' in request.json or not 'requirements' in request.json or not 'model' in request.json:
        abort(400)
    preprocess_file = request.json['preprocess']
    requirements_file = request.json['requirements']
    model_file = request.json['model']
    hdfs_uri = request.json['hdfs_uri']
    logger.info('Read json configurations: OK!!')
    
    # Download files from HDFS
    client_hdfs = InsecureClient(hdfs_uri)
    client_hdfs.download(requirements_file, "./requirements.txt", overwrite=True)
    client_hdfs.download(model_file, "./model.pickle", overwrite=True)
    client_hdfs.download(preprocess_file, "./preprocess.pickle", overwrite=True)
    logger.info('Download pickles: OK!!')
    
    # Install library dependencies
    subprocess.call("pip install -r ./requirements.txt", shell=True)
    current_app.hdfs_uri = hdfs_uri
    current_app.inMemory = False
    current_app.configured = True
    return jsonify({'status': "Docker configured"}), 201


@application.route('/predict', methods=['POST'])
def predict():
    
    # Read inputs/classes or Return reason of abort
    if not hasattr(current_app, 'configured'):
        abort(Response('Docker not configured. Need to call /config API first.'))
    
    if request.json and 'classes' in request.json:
        classes = request.json['classes']
        if 'inputs' in request.json:
            inputs = request.json['inputs']
        elif 'csv_path' in request.json:
            hdfs_uri = request.json['hdfs_uri']
            client_hdfs = InsecureClient(hdfs_uri)
            client_hdfs.download(request.json['csv_path'], "./tweets.csv", overwrite=True)
            inputs = list(pd.read_csv('tweets.csv', names=['tweets'], skiprows=1).tweets)
    else:
        abort(Response('Json not understandable, make sure that you have "inputs" and "classes" key.'))

    
    # load model and preprocess/postprocess in memory
    if (not current_app.inMemory):
        current_app.preprocess = cloudpickle.load(open("./preprocess.pickle", "rb"))
        current_app.finalModel = cloudpickle.load(open("./model.pickle", "rb"))
        current_app.inMemory = True
    logger.info('Load model: OK!!')
    
    # Forward pass for predict
    preproc_inputs = current_app.preprocess(inputs)
    logger.info('pre-processed inputs: ', preproc_inputs)
    outputs = current_app.finalModel(preproc_inputs)
    logger.info('outputs: ', outputs)
    
    # output predicted class
    predictions = np.argmax(outputs.detach().numpy(), axis=1)
    predictions = [classes[pred] for pred in predictions]
    
    return jsonify({'predictions': predictions}), 201


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=2333)
