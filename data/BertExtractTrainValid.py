import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

aml = "aml.csv"
content = "amlcont.csv"

df1 = pd.read_csv(aml)
df2 = pd.read_csv(content)

brokenData = False

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

for idx, lab, cont in zip(df1['index'], df1['label'], df2['content']):
    for name in eval(lab):
        if ',' in name: name = name.replace(',', ' ')
        if name not in cont:
            print(idx, name, 'not in the cont below')
            print(cont)
            brokenData = True

if brokenData: exit(1)

def isTag(e):
    return e[0] == '[' and e[-1] == ']'

def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and \
                (''.join(mylist[i:i+len(pattern)]) == pattern or \
                    any([isTag(e) for e in mylist[i:i+len(pattern)]])):
            matches.append(i)
    return matches

def match_label(toks, labs, cont):

    label = ['X' if t.startswith('##') else 'O' for t in toks]

    mod = False
    for lab in sorted(labs):
        begs = subfinder(toks, lab)
        print(len(begs), lab)
        if not begs: 
            #print('cannot find', lab)
            mod = True
        for beg in begs:
            label[beg] = 'B-per'
            for i in range(len(lab) - 1):
                label[beg + i + 1] = 'I-per'

    if mod:
        #print(cont)
        label = ['U' if isTag(t) else l for l, t in zip(label, toks)]
        for i in range(len(label) - 1):
            if label[i] == 'U':
                beg = i
                label[beg] = 'B-per'
                beg += 1
                while label[beg] == 'U':
                    label[beg] = 'I-per'
                    beg += 1
    #print(toks)
    #print(label)
    return label

index = []
tokens = []
labels = []
maxlen = 0

for idx, labs, cont in zip(df1['index'], df1['label'], df2['content']):
    labs = [name.replace(',', ' ') if ',' in name else name for name in set(eval(labs))]
    for sent in [c.strip() for c in re.split('。', cont.replace('。」', '」').replace('。）', '）'))]:
        toks = tokenizer.tokenize(sent + '。')
        if any([n in sent for n in labs]):
            label = match_label(toks, [l for l in labs if l in sent], cont)
        else:
            label = ['O' for _ in range(len(toks))]
        if len(toks) < 8:
            #print(toks)
            continue
        index.append(idx)
        tokens.append(toks)
        labels.append(label)
        maxlen = max(maxlen, len(label))

print(maxlen)

data = pd.DataFrame({'index': index, 'token': tokens, 'label': labels})
train, valid = train_test_split(list(set(index)), test_size=0.2, random_state = 42)
train = data[[e in train for e in data['index']]]
valid = data[[e in valid for e in data['index']]]
train.to_csv("TrainExtract.csv", index=False)
valid.to_csv("ValidExtract.csv", index=False)
