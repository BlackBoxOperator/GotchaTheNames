#!/usr/bin/env python3.6
# coding: utf-8

from tqdm import *
import numpy as np
import time, os, json, csv, re, sys
import shutil

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import joblib

from gensim import corpora
from gensim.models import Phrases


from bm25 import BM25Transformer
from logger import EpochLogger
from make_token import token_maker, method, usage
from ckip_util import name_extractor
from fit import fitness

###########################
#  modifiable parameters  #
###########################

title_weight = 2
retrieve_size = 300

w2vType = 't2m3w7d100e100'
amlFile = os.path.join('..', 'data', 'aml.csv')

def mapTrim(f):
    return [t.strip() for t in f if t.strip()]

def retain_chinese(line):
    return re.compile(r"[^\u4e00-\u9fa5]").sub('', line).replace('臺', '台')

def get_screen_len(line):
    chlen = len(retain_chinese(line))
    return (len(line) - chlen) + chlen * 2

def get_full_screen_width():
    return shutil.get_terminal_size((80, 20)).columns

def old_models(config):

    models = {}

    print("loading bm25Cache...", end='');
    sys.stdout.flush()
    models['bm25'] = joblib.load(config['bm25Cache'])
    print("ok")
    print("loading docBM25Cache...", end='')
    sys.stdout.flush()
    models['doc_bm25'] = joblib.load(config['docBM25Cache'])
    print("ok")
    print("loading vectorizerCache...", end='')
    sys.stdout.flush()
    models['vectorizer'] = joblib.load(config['vectorizerCache'])
    print("ok")

    print("loading w2v model...", end='')
    sys.stdout.flush()
    models['w2v'] = config['load_w2v']()
    print("ok")
    print("loading docW2VCache...", end='')
    sys.stdout.flush()
    models['docWv'] = joblib.load(config['docW2VCache'])
    print("ok")

    return models

def new_models(config):

    models = {}

    token = mapTrim(open(config['tokenFile'], encoding="UTF-8").read().split('\n'))
    title = mapTrim(open(config['titleFile'], encoding="UTF-8").read().split('\n'))

    if len(config['tokey']) != len(token) or len(token) != len(title):
        print('len(token) {} != len(tokey) {}'.format(len(token), len(config['tokey'])))
        exit(0)

    # append title to doc
    print("\nappending title to document...\n")

    for i, key in enumerate(tqdm(config['tokey'])):
        if title and title != "Non":
            token[i] += ' {}'.format(title[i]) * title_weight

    print("\nbuilding corpus vector space...\n")

    models['bm25'] = BM25Transformer()
    models['vectorizer'] = TfidfVectorizer()
    doc_tf = models['vectorizer'].fit_transform(tqdm(token))

    print("fitting bm25...", end='');
    sys.stdout.flush()
    models['bm25'].fit(doc_tf)
    print("transforming...", end='');
    models['doc_bm25'] = models['bm25'].transform(doc_tf)
    print("ok")

    print("saving bm25Cache...", end='');
    sys.stdout.flush()
    joblib.dump(models['bm25'], config['bm25Cache'])
    print("ok")
    print("saving docBM25Cache...", end='');
    sys.stdout.flush()
    joblib.dump(models['doc_bm25'], config['docBM25Cache'])
    print("ok")
    print("saving vectorizerCache...", end='');
    sys.stdout.flush()
    joblib.dump(models['vectorizer'], config['vectorizerCache'])
    print("ok")

    print('\ncorpus vector space - ok\n')

    docsTokens = [t.split() for t in token]

    # mod
    print("loading w2v model...", end='')
    sys.stdout.flush()
    models['w2v'] = config['load_w2v']()
    print("ok")

    print("making document word vector")

    models['docWv'] = np.array(
            [np.sum(models['w2v'][[t for t in docsTokens[i] if t in models['w2v']]], axis=0) \
                        for i in tqdm(range(len(docsTokens)))])

    print("saving docW2VCache...", end='');
    sys.stdout.flush()
    joblib.dump(models['docWv'], config['docW2VCache'])
    print("ok")
    return models

def load_models(config):

    config['tokey'] = mapTrim(open(config['tokeyFile'], encoding="UTF-8").read().split('\n'))
    if all([os.path.isfile(cache) for cache in config['caches']]):
        return old_models(config)
    else:
        return new_models(config)

def getConfig(name, check_w2v = True, check_token = True,
        tokname = '', model_type = w2vType, pred_w2v = None):
    config = {}

    if tokname: tokname += '_'

    config['tokenFile'] = os.path.join('..', 'token', '{}_{}token.txt'.format(name, tokname))
    config['tokeyFile'] = os.path.join('..', 'token', '{}_{}tokey.txt'.format(name, tokname))
    config['titleFile'] = os.path.join('..', 'token', '{}_{}title.txt'.format(name, tokname))

    if pred_w2v:
        config['w2vFile'] = os.path.splitext(os.path.basename(pred_w2v))[0]
        config['w2vPath'] = pred_w2v
    else:
        config['w2vFile'] = '{}_{}{}'.format(name, tokname, model_type)
        config['w2vPath'] = os.path.join('..', 'w2v', config['w2vFile'] + '.w2v')

    check = [amlFile]

    if check_token:
        check = check + [config['tokenFile'], config['tokeyFile'], config['titleFile']]

    if check_w2v: check.append(config['w2vPath'])

    for fn in check:
        if not os.path.isfile(fn):
            print(fn, '''not found.
use `make [data|token]` or `python make_token.py [mode]` to prepare it.''')
            exit(1)

    bm25Cache = os.path.join('..', 'cache', name + '_bm25.pkl')
    docBM25Cache = os.path.join('..', 'cache', name + '_doc_bm25.pkl')
    vectorizerCache = os.path.join('..', 'cache', name + '_vectorizer.pkl')
    docW2VCache = os.path.join('..', 'cache', config['w2vFile'] + '_doc_w2v.pkl')
    caches = [bm25Cache, docBM25Cache, vectorizerCache, docW2VCache]

    config['bm25Cache']       = bm25Cache
    config['docBM25Cache']    = docBM25Cache
    config['vectorizerCache'] = vectorizerCache
    config['docW2VCache']     = docW2VCache
    config['caches']          = caches
    config['tokenize']        = token_maker(name)
    config['load_w2v']        = method[name]['w2v'](config['w2vPath'])

    return config


def ranking_by_stage(config, models, info, cap, cont, stages):
    query = config['tokenize'](cont)
    query += ' {}'.format(config['tokenize'](cap)) * title_weight

    # w2v
    indexs = [_ for _ in query.split() if _ in models['w2v'].vocab]
    if indexs:
        qryWv = np.sum(
                models['w2v'][indexs],
                axis=0)
        #scores = models['w2v'].cosine_similarities(qryWv, models['docWv'])
    else:
        qryWv = np.zeros((models['w2v'].vector_size,))
        #scores = np.zeros((models['doc_bm25'].shape[0],))

    # bm25
    qryTf = models['vectorizer'].transform([query])
    qryBm25 = models['bm25'].transform(qryTf)

    def ranking(qry_bm25, qry_wv):
        sims = cosine_similarity(qry_bm25, models['doc_bm25']).reshape(-1)
        sims += models['w2v'].cosine_similarities(qry_wv, models['docWv'])
        return sorted(zip(config['tokey'], sims),
                    key=lambda e: e[-1], reverse=True)

    def feedback(pre_qrybm25, pre_qryWv, rks, n):
        weight = lambda rk: 0.5
        rk2docBm = lambda rk: models['doc_bm25'][config['tokey'].index(rks[rk][0])]
        rk2docWv = lambda rk: models['docWv'][config['tokey'].index(rks[rk][0])]

        new_qrybm25 = pre_qrybm25 + np.sum(rk2docBm(rk) * weight(rk) for rk in range(n))

        #new_qryWv = pre_qryWv # previous impl
        new_qryWv = pre_qryWv + np.sum(rk2docWv(rk) * weight(rk) for rk in range(n))

        return new_qrybm25, new_qryWv
        #np.sum(np.fromiter(models ... ))) # for 3.7

    for stage, n in enumerate(stages):
        print("\033[F" + info.format(stage + 1))
        ranks = ranking(qryBm25, qryWv)
        qryBm25, qryWv = feedback(qryBm25, qryWv, ranks, n)

    return ranking(qryBm25, qryWv)

def exist_english(s):
    return any([c.isalpha() for c in s])

def aml_name_extractor(name):

    config = getConfig(name)
    models = load_models(config)
    aml_list = [id for id, *_ in list(csv.reader(open(amlFile, encoding="UTF-8")))[1:]]
    extract_name = name_extractor()

    stages = [20, 40, 60, 100]
    def extract(doc):
        threshold = 50
        info = '[ stage {{}}/{} ]'.format(len(stages))
        ranks = ranking_by_stage(config, models, info, '', doc, stages)
        ranks = ranks[:retrieve_size] # cut ranks
        hit_count = sum([1 for id, _ in ranks if id in aml_list])
        hit_score = [s for id, s in ranks if id in aml_list]
        names, secs = extract_name(doc)[0] if hit_count > 50 else ([], 0)

        # remove space (if any) in chinese name
        names = [''.join(n.split()) if exist_english(n) else n for n in names]

        # remove substring
        names = [this for idx, this in enumerate(names)
                if not any(this in other for other in names[idx + 1:])]

        return names, secs

    return extract

def keyword_tuner(table, queryFile = os.path.join('..', 'query', 'query.csv')):

    name = "search_big"
    config = getConfig(name)
    #models = load_models(config)
    #print("query file: {}\nw2vFile: {}".format(queryFile, config['w2vPath']))
    aml_list = [id for id, *_ in list(csv.reader(open(amlFile, encoding="UTF-8")))[1:]]
    q_tokens = []
    keywords = set(x for x in table.keys())
    queries = list(csv.reader(open(queryFile, 'r', encoding="UTF-8")))
    for idx, [q_id, q_cap, q_cont] in enumerate(tqdm(queries)):
        tokens = config['tokenize'](q_cont).split()
        q_tokens.append(
                ('v' if q_id in aml_list else 'x',
                    len(tokens), [tok for tok in tokens if tok in keywords]))

    def tuning_by_keyword(table):

        hit_count = [0 for _ in range(len(queries))]
        hit_score = [[] for _ in range(len(queries))]
        d = {'v': [], 'x': []}

        for l, doclen, tokens in q_tokens:
            val = 0
            for token in tokens:
                if token in table:
                    val += table[token]
            d[l].append(val / doclen)

        return fitness(d['v'], d['x'], retrieve_size)

    return tuning_by_keyword



if __name__ == '__main__':

    if len(sys.argv) < 2:
        usage('[query.csv] [corpus] [predict.w2v]')
        exit(0)
    else:
        name = sys.argv[1]

    if name not in method:
        print("no such mode")
        usage('[query.csv] [corpus] [predict.w2v]')
        exit(1)

    qname = ''
    pred_w2v = None
    queryFile = os.path.join('..', 'query', 'query.csv')

    tokname = ''
    for arg in sys.argv[2:]:
        bname, ext = os.path.splitext(os.path.basename(arg))
        if ext == '.csv':
            queryFile = arg
            qname = os.path.splitext(os.path.basename(queryFile))[0] + '_'
        elif ext == '.w2v':
            pred_w2v = arg
        else:
            tokname = arg

    if not pred_w2v:
        if name == 'search':
            model_type = 't2m5w5d100e100'
        elif name == 'search_big':
            model_type = 't2m3w7d100e100'
        else:
            model_type = w2vType

    config = getConfig(name, model_type = model_type, tokname = tokname, pred_w2v = pred_w2v)
    models = load_models(config)

    score_result = os.path.join('{}{}_score.txt'.format(qname, config['w2vFile']))
    rank_result = os.path.join('{}{}_submit.csv'.format(qname, config['w2vFile']))

    print("query file: {}\ncorpus file: {}\nw2vFile: {}".format(
        queryFile, config['tokenFile'], config['w2vPath']))

    aml_list = [id for id, *_ in list(csv.reader(open(amlFile, encoding="UTF-8")))[1:]]

    showLastQueryProgress = False
    recordRanks = False

    score_file = open(score_result, 'w', newline='', encoding="UTF-8")

    if recordRanks:
        csvfile = open(rank_result, 'w', newline='', encoding="UTF-8")
        writer = csv.writer(csvfile)
        headers = ['Query_Index'] + ['Rank_{:03d}'.format(i) for i in range(1, retrieve_size + 1)]
        writer.writerow(headers)

    queries = list(csv.reader(open(queryFile, 'r', encoding="UTF-8")))
    hit_count = [0 for _ in range(len(queries))]
    hit_score = [[] for _ in range(len(queries))]

    for idx, [q_id, q_cap, q_cont, *_] in enumerate(tqdm(queries)):

        stages = [20, 40, 60, 100]

        info = '[ stage {{}}/{} ]'.format(len(stages))

        print('{} Query{}: {}'.format(
            info.format(0), idx + 1,
            q_cap[:min(30, (get_full_screen_width() // 4))]))


        #ranks = ranking_by_stage(config, models, info, q_cap, q_cont, stages)
        ranks = ranking_by_stage(config, models, info, '', q_cont, stages)

        ranks = ranks[:retrieve_size] # cut ranks

        hit_count[idx] = sum([1 for id, _ in ranks if id in aml_list])
        hit_score[idx] = [s for id, s in ranks if id in aml_list]

        if showLastQueryProgress and idx == len(queries) - 1:
            print("\033[F" * 3)
            print("\033[B", end='')
        else:
            print("\033[F" + ' ' * get_full_screen_width())
            print("\033[F" * 3)

        if recordRanks: writer.writerow([q_id] + [e[0] for e in ranks])

        line = '[ {} {:3d} / {} = {:4f} score = {:6.2f} ] Query{}: {}'.format(
            'v' if q_id in aml_list else 'x',
            hit_count[idx],
            retrieve_size,
            hit_count[idx] / retrieve_size,
            sum(hit_score[idx]),
            idx + 1, q_cap[:30])

        score_file.write(line + '\n')
        if idx % 100 == 0: score_file.flush()

    if recordRanks: csvfile.close()
    score_file.close()
    exit(0) # not show

    print('-' * get_full_screen_width())
    with open(score_result, 'w', newline='', encoding="UTF-8") as score_file:
        for idx, [q_id, q_cap, q_cont] in enumerate(queries):
            line = '{} [ {:3d}/{} = {:4f}, score = {:6.2f} ] Query{}: {}'.format(
                'v' if q_id in aml_list else 'x',
                hit_count[idx],
                retrieve_size,
                hit_count[idx] / retrieve_size,
                sum(hit_score[idx]),
                idx + 1, q_cap[:30])
            print(line)
            score_file.write(line + '\n')
            #score_file.write('\n'.join([str(s) for s in hit_score[idx]]) + '\n')
