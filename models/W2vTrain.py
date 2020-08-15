from tqdm import *
import numpy as np
import os, datetime, re, sys
from gensim import corpora
from gensim.models import Phrases
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from logger import EpochLogger

from oracle import getConfig, mapTrim
from make_token import usage, method

dim = 100
min_count = 3
window = 7
epoch = 100
title_weight = 2

if len(sys.argv) < 2:
    usage('[content.csv]')
    exit(0)
else:
    name = sys.argv[1]

if name not in method:
    print("no such mode")
    usage('[content.csv]')
    exit(1)

if len(sys.argv) >= 3:
    content = sys.argv[2]
    cname = os.path.splitext(os.path.basename(content))[0]
else:
    content = os.path.join('..', 'data', 'content.csv')
    cname = ''

model_type =  't{}m{}w{}d{}e{}'.format(title_weight, min_count, window, dim, epoch)
config = getConfig(name, check_w2v = False, tokname = cname, model_type = model_type)
model_name = config['w2vPath']

print("opening file...")

tokey = mapTrim(open(config['tokeyFile'], encoding="UTF-8").read().split('\n'))
title = mapTrim(open(config['titleFile'], encoding="UTF-8").read().split('\n'))
token = mapTrim(open(config['tokenFile'], encoding="UTF-8").read().split('\n'))

if len(tokey) != len(token) or len(token) != len(title):
    print('len(token) {} != len(tokey) {}'.format(len(token), len(tokey)))
    exit(0)

for i, key in enumerate(tqdm(tokey)):
    if title and title != "Non":
        token[i] += ' {}'.format(title[i]) * title_weight

print("""
appending title to document...
""")

print("spliting tokens...")
tokens = [doc.split() for doc in tqdm(token)]

print("creating model...")
model = Word2Vec(
        tqdm(tokens),
        min_count=min_count,
        size=dim,
        window=window,
        workers=3,
        callbacks=[EpochLogger()])

print("training model...")
model.train(tqdm(tokens), total_examples=len(tokens), epochs=epoch, callbacks=[EpochLogger()])

print("saving model...")
model.save(model_name)
