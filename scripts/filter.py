import os, csv, sys

byId = True
byIdx = False

pred_file = sys.argv[1]
corpus_file= sys.argv[2]
#pred_file = os.path.join('..', 'data', 'aml.csv')
#corpus_file = os.path.join('..', 'data', 'content.csv')

pred = csv.reader(open(os.path.join(pred_file), 'r'))
next(pred)
pred = set(int(r) for r, *_ in pred)

csv.field_size_limit(sys.maxsize)
corpus = csv.reader(open(os.path.join(corpus_file), 'r'))
next(corpus)

with open('data.csv', 'w', newline='', encoding='UTF-8') as csvfile:
    writer = csv.writer(csvfile)
    for idx, (id, title, cont) in enumerate(corpus):
        if byId:
            if int(id) not in pred:
                writer.writerow([id, title, cont])
        if byIdx:
            if idx + 1 in pred:
                writer.writerow([id, title, cont])
