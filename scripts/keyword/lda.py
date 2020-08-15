"""
https://www.jianshu.com/p/a796ca559409
"""
import os
import jieba.analyse as analyse
import jieba
import pandas as pd
from gensim import corpora, models, similarities
import gensim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
#%matplotlib inline


num_topics  = 8
num_show_term = 2500

sentences = []

def mapTrim(f):
    return [t.strip() for t in f if t.strip()]

#tokeyFile = os.path.join("..", "..", "token", "search_big_keyword_tokey.txt")
titleFile = os.path.join("..", "..", "token", "search_big_keyword_title.txt")
tokenFile = os.path.join("..", "..", "token", "search_big_keyword_token.txt")
#titleFile = os.path.join("..", "..", "token", "search_big_title.txt")
#tokenFile = os.path.join("..", "..", "token", "search_big_token.txt")
#tokey = mapTrim(open(tokeyFile, encoding="UTF-8").read().split('\n'))
title = mapTrim(open(titleFile, encoding="UTF-8").read().split('\n'))
token = mapTrim(open(tokenFile, encoding="UTF-8").read().split('\n'))

for t, d in zip(title, token):
    sentences.append((t + ' ' + d).split())

dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)


keywords = set()
for topic, words in lda.show_topics(num_topics=10, num_words=100, log=False, formatted=False):
    for word, prob in words:
        keywords.add(word)

with open('key_lda.txt', 'w', newline='', encoding='UTF-8') as keyfile:
    for word in keywords:
        keyfile.write(word + '\n')



showGraph = False
if showGraph:
    fontPath = r'NotoSansMonoCJKtc-Bold.otf'
    font30 = fm.FontProperties(fname=fontPath, size=30)

    for i, k in enumerate(range(num_topics)):
        ax = plt.subplot(2, 5, i+1)
        item_dis_all = lda.get_topic_terms(topicid=k)
        item_dis = np.array(item_dis_all[:num_show_term])
        ax.plot(range(num_show_term), item_dis[:, 1], 'b*')
        item_word_id = item_dis[:, 0].astype(np.int)
        word = [dictionary.id2token[i] for i in item_word_id]
        ax.set_ylabel('probability', fontproperties=font30, fontsize=8)
        for j in range(num_show_term):
            ax.text(j, item_dis[j, 1], word[j], bbox=dict(facecolor='green',alpha=0.1), fontproperties=font30, fontsize=8)
    plt.suptitle('{} topic and {} word probability'.format(num_topics, num_show_term), fontsize=12, fontproperties=font30)
    plt.show()
