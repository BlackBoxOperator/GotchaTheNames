import os, csv, sys

meta_file = os.path.join('..', '..', 'data', 'tbrain_train_final_0610.csv')
corpus_file = os.path.join('..', '..', 'data', 'content.csv')
meta = csv.reader(open(os.path.join(meta_file), 'r')); next(meta, None)

target = []
mapping = {}
for id, link, cont, names in meta:
    names = eval(names)
    mapping[id] = link
    if names:
        print(id, names)
        target.append(id)

csv.field_size_limit(sys.maxsize)
corpus = csv.reader(open(os.path.join(corpus_file), 'r'))

for id, title, cont in corpus:
    if not cont and id in target:
        print('{},{}'.format(id, mapping[id]))
    elif not cont:
        pass
        #print("{} {} missing", id, title)
