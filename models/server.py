
import os
from flask import Flask
from flask import request
from flask import jsonify
import datetime
import hashlib
import numpy as np
import pandas as pd
from aml import aml_name_extractor
import logging
from pprint import pprint

from time import sleep

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'your@mail.com'#
SALT = 'BlackboxOperators'              #
#########################################

amlex = aml_name_extractor("search_big")

logpath = os.path.join('log',
        str(datetime.datetime.now().utcnow()).replace(' ', '_') + '.log')

logfile = open(logpath, 'w')

def log(*args):
    print(*args, file = logfile)
    print('', file = logfile)
    logfile.flush()

def get_timestamp():
    return (datetime.datetime.now().utcnow().timestamp())

def generate_server_uuid(input_string):
    """ Create your own server_uuid
    @param input_string (str): information to be encoded as server_uuid
    @returns server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string+SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid

def predict(article):
    """ Predict your model result
    @param article (str): a news article
    @returns prediction (list): a list of name
    """
    global amlex

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######
    prediction = amlex(article)

    ####################################################
    prediction = _check_datatype_to_list(prediction)
    return prediction

def _check_datatype_to_list(prediction):
    """ Check if your prediction is in list type or not.
        And then convert your prediction to list type or raise error.

    @param prediction (list / numpy array / pandas DataFrame): your prediction
    @returns prediction (list): your prediction in list type
    """
    if isinstance(prediction, np.ndarray):
        _check_datatype_to_list(prediction.tolist())
    elif isinstance(prediction, pd.core.frame.DataFrame):
        _check_datatype_to_list(prediction.values)
    elif isinstance(prediction, list):
        return prediction
    raise ValueError('Prediction is not in list type.')

@app.route('/healthcheck', methods=['POST'])
def healthcheck():
    """ API for health check """
    data = request.get_json(force=True)
    print(data)
    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)
    server_timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'esun_uuid': data['esun_uuid'], 'server_uuid': server_uuid, 'captain_email': CAPTAIN_EMAIL, 'server_timestamp': server_timestamp})

cache_answer = {}

@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API """
    data = request.get_json(force=True)
    esun_timestamp = data['esun_timestamp']
    server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ts = str(int(datetime.datetime.now().utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)

    answer_template = lambda ans: jsonify({
            'esun_timestamp': data['esun_timestamp'],
            'server_uuid': server_uuid,
            'answer': ans,
            'server_timestamp': server_timestamp,
            'esun_uuid': data['esun_uuid']
            })

    if data['esun_uuid'] in cache_answer:
        if cache_answer[data['esun_uuid']] != None:
            return answer_template(cache_answer[data['esun_uuid']])
        else:
            while cache_answer[data['esun_uuid']] == None:
                sleep(4)
            return answer_template(cache_answer[data['esun_uuid']])
    else:
        cache_answer[data['esun_uuid']] = None
        try:
            log(data['news'])
            answer = predict(data['news'])
            log(answer)
        except:
            log('model error')
            raise ValueError('Model error.')

        cache_answer[data['esun_uuid']] = answer

        return answer_template(answer)

@app.route('/')
def homepage():
    return 'Welcome to AML target detection system'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)

"""
# useful reference
https://datahunter.org/ssh_tunnel

# reverse agent
ssh -R :8080:localhost:8080 zxc@lab
ssh -R :80:localhost:8080 user044@aml

# forward 80 port 8080 port
git clone https://github.com/vinodpandey/python-port-forward.git
sudo python2.7 port-forward.py 80:localhost:8080

# check the binded port and kill
sudo lsof -wni tcp:8080

autossh -M 20000 -i ~/.ssh/id_rsa -NfR  :8080:localhost:8080 zxc@lab
"""
