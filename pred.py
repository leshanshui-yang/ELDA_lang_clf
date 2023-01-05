import os
import pandas as pd
import numpy as np  
import torch
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer

from flask import Flask, jsonify, request
from flask import current_app, abort, Response

import warnings, logging
import subprocess
import requests


#%%
### Logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.ERROR)

### Init App
application = Flask(__name__)
application.logger.addHandler(handler)

### Params
classes = {0:"Negative", 1:"Positive"}
device = "cpu"
model_name = 'prajjwal1/bert-tiny'
model_pt_path = 'model/tinybert_imdb_230105.pth'

### Loading Model and Tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
state_dict = torch.load(model_pt_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)


@application.route('/predict', methods=['POST'])
def predict():
    # Read inputs/classes or Return reason of abort
    if 'inputs' in request.json:
        texts = request.json['inputs']
    else:
        abort(Response('Json not understandable, make sure that you have "inputs" key.'))
        
    ### Preprocessing
    # Tokenization (code was designed for batch inference, but in real case we have only 1 text in texts list)
    inputs_mask = [] 
    texts_ids = []
    for text in texts: 
        text_tokenized = tokenizer.tokenize(text.replace('[^\w\s]', ''))
        text_ids = tokenizer.convert_tokens_to_ids([tokenizer.special_tokens_map["cls_token"]]+text_tokenized+[tokenizer.special_tokens_map["sep_token"]])
        texts_ids.append(text_ids)
        inputs_mask.append([1] * len(text_ids))
    # Batch-Padding (code was designed for batch inference, but in real case we have only 1 text in texts list)
    lengths = [len(p) for p in texts_ids]
    max_length = max(lengths)
    padded_text_ids = np.ones((len(texts_ids), max_length))*tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map["pad_token"])
    padded_inputs_mask = np.zeros((len(texts_ids), max_length))
    for i, l in enumerate(lengths):
        padded_text_ids[i, 0:l] = texts_ids[i][0:l]
        padded_inputs_mask[i, 0:l] = inputs_mask[i][0:l]
    # Transpose to 3 tensors to adapt the input of inference()
    padded_tensors = torch.LongTensor(np.array([padded_text_ids, padded_inputs_mask]))
    
    ### Prediction
    text_ids, inputs_mask = padded_tensors
    outputs = model(text_ids.to(device), inputs_mask.to(device))
    predictions = np.argmax(outputs.logits.detach().cpu().numpy(), axis=-1)
    predictions = [classes[pred] for pred in predictions]
    return jsonify({'predictions': predictions}), 201


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8080)
    
