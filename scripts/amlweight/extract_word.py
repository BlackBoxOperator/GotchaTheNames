import sys, os
from gensim.models import Word2Vec
from logger import EpochLogger

def mapTrim(f):
    return [t.strip() for t in f if t.strip()]

modelFile = os.path.join("..", "..", "w2v", "search_big_t2m3w7d100e100.w2v")

keywords = set()
print("loading model")
model = Word2Vec.load(modelFile)
print("loading model done")

tokeyFile = os.path.join("..", "..", "token", "search_big_amlcont_tokey.txt")
titleFile = os.path.join("..", "..", "token", "search_big_amlcont_title.txt")
tokenFile = os.path.join("..", "..", "token", "search_big_amlcont_token.txt")

tokey = mapTrim(open(tokeyFile, encoding="UTF-8").read().split('\n'))
title = mapTrim(open(titleFile, encoding="UTF-8").read().split('\n'))
token = mapTrim(open(tokenFile, encoding="UTF-8").read().split('\n'))

for doc in title:
    for word in doc.split():
        if word in model.wv.vocab:
            keywords.add(word)

for doc in token:
    for word in doc.split():
        if word in model.wv.vocab:
            keywords.add(word)

print(len(keywords))
