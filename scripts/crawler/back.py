import csv
from crawler import fetch_news, eprint, writerow, mode

missurls = open('miss.txt').read().split()
misspath = open('path.txt').read().split()

def record(a):
    record.data = a

record.data = []
miss = 0

#with open('log.csv', mode, encoding="UTF-8") as logfile:
#    log = lambda *ss: logfile.write(' '.join([str(s) for s in ss]) + '\n')
#    with open('ret.csv', mode, newline='', encoding="UTF-8") as csvfile:
#        writer = csv.writer(csvfile)
#        writerow(writer, ['index', 'title', 'content'], log)

for index, url in [x.split(',') for x in missurls]:
    paths = [p for p in misspath if url in p]
    find = False
    for p in paths:
        record.data = []
        fetch_news(index, p, record, lambda *a: 2)
        [idx, title, cont] = record.data
        if title and cont:
            find = True
            break
    if not find:
        miss += 1
        print(index, 'not found')
        for p in paths:
            print(p)
    else:
        pass
        #writerow(writer, record.data, log)

print('total miss', miss)
