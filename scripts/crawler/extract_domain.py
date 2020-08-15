import os, csv
from pprint import pprint

def extract_domain(url):
    url = url[url.index("://") + len("://"):]
    domain = url[:url.index('/') if '/' in url else len(url)]
    return domain

mapping = {}
ctx = open(os.path.join('..', '..', 'data', 'tbrain_train_final_0610.csv'), 'r')
csvr = csv.reader(ctx); next(csvr, None)
nc = {row[0]: row[1] for row in csvr}

for iden in nc:
    if "://" in nc[iden]:
        domain = extract_domain(nc[iden])
        mapping.setdefault(domain, []).append(nc[iden])

pprint(mapping)
for k in mapping.keys():
    print(k)
    print(mapping[k][0])
