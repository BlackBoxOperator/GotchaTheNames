import sys
from gensim.models import Word2Vec
from logger import EpochLogger

modelFile = sys.argv[1]

print("loading model")
model = Word2Vec.load(modelFile)
print("loading model done")

while True:
    print(model.wv.most_similar(positive=input('> ').split(), topn=10))
