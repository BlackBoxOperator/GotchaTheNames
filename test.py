import os, sys, csv
import requests
import hashlib
import datetime
import pandas as pd
from pprint import pprint
from time import sleep

ip = 'http://127.0.0.1:8080'

time_limit = 5

SALT='BlackboxOperator'
def generate_task_uuid(input_string):
    print(input_string)
    s = hashlib.sha256()
    data = (input_string+SALT).encode("utf-8")
    s.update(data)
    return s.hexdigest()

def get_timestamp():
    return int(datetime.datetime.now().utcnow().timestamp())

def question(news):

    task_uuid = generate_task_uuid(str(datetime.datetime.now().utcnow().timestamp()))

    print('task_uuid:', task_uuid)

    for retry in range(2, -1, -1):
        check_data = {
            "esun_uuid" : task_uuid,
            "esun_timestamp" : get_timestamp(),
            "retry" : retry
        }
        try:
            hr = None
            hr = requests.post('{}/healthcheck'.format(ip),
                    json = check_data,
                    timeout = time_limit)
        except Exception as e:
            print(e)
        if hr and hr.status_code == requests.codes.ok:
            break
    else: raise Exception("err, health check failed")

    print("pass health check")

    for retry in range(2, -1, -1):
        stmp = get_timestamp()
        inference_data = {
            "esun_uuid" : task_uuid,
            "server_uuid" : hr.json()['server_uuid'],
            "esun_timestamp" : stmp,
            "news" : news,
            "retry" : retry
        }
        try:
            ir = None
            ir = requests.post('{}/inference'.format(ip),
                    json = inference_data,
                    timeout = time_limit)
            print(ir.json()['answer'])
            print("total cost {} secs.".format(get_timestamp() - stmp))
        except Exception as e:
            print(e)
        if ir and ir.status_code == requests.codes.ok:
            break
    else: raise Exception("inference failed")

    return ir.json()['answer']

def recall(tr, pr):
    return len(set(tr) & set(pr)) / len(tr)

def precision(tr, pr):
    return len(set(tr) & set(pr)) / len(pr) if len(pr) else 0.0

def f1(tr, pr):
    rc = recall(tr, pr)
    pc = precision(tr, pr)
    return 2 / ((rc ** -1) + (pc ** -1)) if rc and pc else 0.0

def scoring(tr, pr):
    s = []
    for t, p in zip(tr, pr):
        if not isinstance(tr, list) or not isinstance(pr, list):
            s.append(0.0)
        elif not t and not p:
            s.append(1.0)
        elif not t and p:
            s.append(0.0)
        elif t and not p:
            s.append(0.0)
        else:
            s.append(f1(t, p))
    return s

df = pd.read_csv(os.path.join('data', 'aml.csv'))
am = {idx: eval(names) for idx, names in zip(df['index'], df['label'])}
index, truth, predict = [], [], []

if __name__ == '__main__':
    queryFile = sys.argv[1] if sys.argv[1:] else os.path.join('query', 'query.csv')
    queries = pd.read_csv(queryFile)

    for idx, (q_id, q_cap, q_cont, *_) in \
            enumerate(zip(queries['index'],queries['title'], queries['content'])):
        print('Query{}: {}'.format(idx, q_cap))
        index.append(q_id)
        truth.append(sorted(am.get(q_id, [])))
        try:
            answer = question(q_cont)
        except Exception as e:
            print(q_id, e)
            answer = [str(e)]
        predict.append(sorted(answer))
        sleep(0.2)

    result = open("result.txt", "w")
    score = scoring(truth, predict)
    result.write('question: {} score: {}\n'.format(len(score), sum(score)))
    result.write('index,score,pred,truth\n')
    for idx, sc, tr, pr in zip(index, score, truth, predict):
        result.write('{},{},{},{}\n'.format(idx, sc, pr, tr))
    result.close()
    print('score: {}'.format(sum(score)))
