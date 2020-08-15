from tqdm import tqdm
import jieba.analyse
import csv, os

jieba.set_dictionary(os.path.join('..', '..', 'data', 'dict', 'dict.txt.big'))

def mapTrim(f):
    return [t.strip() for t in f if t.strip()]

keywords = set()


extractA = lambda doc: jieba.analyse.extract_tags(
                doc ,topK=20, withWeight=False, allowPOS=())

extractB = lambda doc: jieba.analyse.extract_tags(
                doc, topK=10, withWeight=False, allowPOS=(['n','v']))

corpus_file = os.path.join('..', '..', 'data', 'amlcont.csv')
#corpus_file = os.path.join('..', '..', 'data', 'keyword.csv')
corpus = csv.reader(open(os.path.join(corpus_file), 'r'))
next(corpus, None)

for id, title, cont in tqdm(list(corpus)):
    doc = title + ' ' + cont
    keywords.update(extractA(doc))
    keywords.update(extractB(doc))

with open('key_tfidf.txt', 'w', newline='', encoding='UTF-8') as keyfile:
    for word in keywords:
        keyfile.write(word + '\n')
