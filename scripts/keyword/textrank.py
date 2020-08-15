from tqdm import tqdm
import jieba.analyse
import csv, os

jieba.set_dictionary(os.path.join('..', '..', 'data', 'dict', 'dict.txt.big'))

def mapTrim(f):
    return [t.strip() for t in f if t.strip()]

keywords = set()

extractA = lambda doc: jieba.analyse.textrank(doc, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
extractB = lambda doc: jieba.analyse.textrank(doc, topK=20, withWeight=False, allowPOS=('n','v'))
extractC = lambda doc: jieba.analyse.textrank(doc, topK=20, withWeight=False, allowPOS=())

corpus_file = os.path.join('..', '..', 'data', 'keyword.csv')
corpus = csv.reader(open(os.path.join(corpus_file), 'r'))

for id, title, cont in tqdm(list(corpus)):
    doc = title + ' ' + cont
    keywords.update(extractA(doc))
    keywords.update(extractB(doc))
    keywords.update(extractC(doc))

with open('key_textrank.txt', 'w', newline='', encoding='UTF-8') as keyfile:
    for word in keywords:
        keyfile.write(word + '\n')

"""
from pyhanlp import *
result = HanLP.extractKeyword(doc, 20)
print(result)
"""
