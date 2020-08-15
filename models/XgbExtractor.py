#!/usr/bin/env python3.6
# coding: utf-8

from tqdm import *
import numpy as np
import time, os, json, csv, re, sys
import shutil

import timeit

from pprint import pprint

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.svm import SVC
import xgboost as xgb

import joblib

from gensim import corpora
from gensim.models import Phrases


from bm25 import BM25Transformer
from logger import EpochLogger
from make_token import token_maker, method, usage
#from XLNetExtractor import extractor as main_name_extractor
#from RobertaExtractor import extractor as main_name_extractor # worse
from BertExtractor import extractor as main_name_extractor
from ckip_util import name_extractor as sub_name_extractor
from fit import fitness

from namefilter import namefilter, all_english

""" reference
https://zhuanlan.zhihu.com/p/50657430
"""

"""
clf = LogisticRegression(C=1.0,solver='lbfgs',multi_class='multinomial')
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)
"""

"""
clf = SVC(C=1.0, probability=True) # since we need probabilities
clf.fit(xtrain_svd_scl, ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)
"""

"""
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_tfv.tocsc(), ytrain)
predictions = clf.predict_proba(xvalid_tfv.tocsc())
"""

"""
clf = xgb.XGBClassifier(nthread=10)
clf.fit(xtrain_svd_scl, ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)
"""

###########################
#  modifiable parameters  #
###########################

byW2v = False
title_weight = 1
retrieve_size = 300

w2vType = 't2m3w7d100e100'
amlFile = os.path.join('..', 'data', 'aml.csv')

#namelist = 'ckipnamelist.csv'
#namelist = 'bertnamelist.csv'
namelist = 'allnamelist.csv'

def mapTrim(f):
    return [t.strip() for t in f if t.strip()]

def retain_chinese(line):
    return re.compile(r"[^\u4e00-\u9fa5]").sub('', line).replace('臺', '台')

def get_screen_len(line):
    chlen = len(retain_chinese(line))
    return (len(line) - chlen) + chlen * 2

def get_full_screen_width():
    return shutil.get_terminal_size((80, 20)).columns

def get_indexes(lst, val):
    #return [idx for idx, v in enumerate(lst) if v == val]
    #return [idx for idx, v in enumerate(lst) if v[0] == val[0]]
    return [idx for idx, v in enumerate(lst) \
            if val in ''.join(lst[idx:idx + 3]) and val[0] in v]
    #cc = lambda i, l: lst[i] + lst[i + 1] if i + 1 < len(lst) else lst[i]
    #return [idx for idx, v in enumerate(lst) if val in cc(idx, lst)]

def isReporter(name, cont):
    idxes = get_indexes(cont, name)
    for idx in idxes:
        beg = lambda index, offset: max(0, index - offset)
        end = lambda index, offset: min(index + offset, len(cont))
        if '記者' in cont[beg(idx, 2):end(idx, 0)]:
            #print(cont[beg(idx, 2):end(idx, 0) + 1])
            #print('記者', name, 'idx', idx, cont[idx])
            return True
    return False

def load_data(config, split = False):

    data = {'config': config}

    token = mapTrim(open(config['tokenFile'], encoding="UTF-8").read().split('\n'))
    title = mapTrim(open(config['titleFile'], encoding="UTF-8").read().split('\n'))
    tokey = mapTrim(open(config['tokeyFile'], encoding="UTF-8").read().split('\n'))


    if len(tokey) != len(token) or len(token) != len(title):
        print('len(token) {} != len(tokey) {}'.format(
            len(token), len(tokey)))
        exit(0)

    data['corpus'] = {qidx: (title, cont) for qidx, cont in zip(tokey, token)}
    dataset = list(csv.reader(open(namelist, encoding="UTF-8")))[1:]

    dataset = [[id, eval(acc), eval(pred)] for id, acc, pred in dataset]

    if split:
        #train, valid = dataset[:280], dataset[280:]
        train, valid = train_test_split(dataset, test_size=0.3, random_state = 42)
    else:
        train, valid = dataset, []

    data['TrainData'] = train
    data['ValidData'] = valid

    data['TrainMiss'] = []
    data['ValidMiss'] = []
    for idx, acc, pred in data['TrainData']:
        data['TrainMiss'].extend([n for n in acc if n not in pred])
    for idx, acc, pred in data['ValidData']:
        data['ValidMiss'].extend([n for n in acc if n not in pred])

    return data

def get_config(name, train, tokname = ''):
    config = {}

    if tokname: tokname += '_'

    config['w2vFile'] = '{}_{}{}'.format(name, tokname, w2vType)
    config['w2vPath'] = os.path.join('..', 'w2v', config['w2vFile'] + '.w2v')
    print('w2v path:', config['w2vPath'])

    config['tokeyFile'] = os.path.join('..', 'token', '{}_{}_tokey.txt'.format(name, train))
    config['titleFile'] = os.path.join('..', 'token', '{}_{}_title.txt'.format(name, train))
    config['tokenFile'] = os.path.join('..', 'token', '{}_{}_token.txt'.format(name, train))

    print(config['tokenFile'])
    config['load_w2v']        = method['search']['w2v'](config['w2vPath'])

    return config

def multiclass_logloss(actual, predicted, eps=1e-15):
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

""" append title to doc
title = [title for title, _ in
        [data['corpus'][idx] for idx, *_ in data['TrainData']]]
print("\nappending title to document...\n")
for i, key in enumerate(tqdm(tokey)):
    if title and title != "Non":
        token[i] += ' {}'.format(title[i]) * title_weight
print("\nbuilding corpus vector space...\n")
"""

def xgboost_model(data, amlclf=None):

    model = {'data' : data}

    if byW2v:
        model['w2v'] = config['load_w2v']()
    else:
        token = [cont for _, cont in
                [data['corpus'][idx] for idx, *_ in data['TrainData']]]

        # we can append title to doc, but ignore
        # because we want to retain the setence struct,
        # we don't remove stopword in token, we remove them here
        #stopwordTxt = os.path.join('..', 'data', 'stopword', "stopword.txt")
        #data['stop'] = set(open(stopwordTxt, "r").read().split())
        #token = [' '.join([t for t in doc.split() if t not in data['stop']]) for doc in token]


        model['bm25'] = BM25Transformer()
        model['vectorizer'] = TfidfVectorizer()

        model['vectorizer'].fit(token)
        TrainTf = model['vectorizer'].transform(tqdm(token))
        print("fitting bm25...", end='');
        sys.stdout.flush()
        model['bm25'].fit(TrainTf)
        print("transforming...", end='');
        model['TrainBm25'] = model['bm25'].transform(TrainTf)
        print("ok")

        if amlclf:
            vectorizer = amlclf['vectorizer']
            bm25_model = amlclf['bm25']
        else:
            vectorizer = model['vectorizer']
            bm25_model = model['bm25']

    def fetch_setence_by_winsize(cont, idx, size):
        beg = lambda index, offset: max(0, index - offset)
        end = lambda index, offset: min(idx + offset, len(cont))
        return cont[beg(idx, size):idx] + cont[idx + 1:end(idx, size) + 1]

    def fetch_setence_by_punctuation(cont, idx):
        beg, end = idx, idx
        while cont[beg] != '。' and beg > 0: beg -= 1
        while cont[end] != '。' and end < len(cont) - 1: end += 1
        #print(''.join(cont[beg + 1:end]))
        #beg = max(idx - 7, beg)
        #end = min(idx + 7, end)
        return cont[beg + 1:end]

    model['namefilter'] = namefilter()
    model['reporter'] = set()

    def names_to_bm25(names, cont):

        names = model['namefilter'](names, cont)
        # consider remove stop word
        cont = cont.split()
        #cont = [t for t in cont if t not in data['stop']]

        people, description = [], []

        model['reporter'] |= set([name for name in names if isReporter(name, cont)])

        names = [name for name in names if not isReporter(name, cont)]

        descs_of_name = lambda name, cont: \
                [fetch_setence_by_winsize(cont, idx, 5) for idx in get_indexes(cont, name)]
                #[fetch_setence_by_punctuation(cont, idx) for idx in get_indexes(cont, name)]

        #descs = {name: descs_of_name(name, cont) for name in names}
        name_descs = [(name, descs_of_name(name, cont)) for name in names]
        descs = {nm: dscs for nm, dscs in name_descs if dscs}

        if len(descs) == 0:
            print(names)
            print(name_descs)
            print(cont)
            return [], []

        if amlclf: descs = {
                name: [amlclf['data']['config']['tokenize'](''.join(desc)).split() \
                            for desc in descs[name]] for name in descs}

        if byW2v: base = np.zeros((model['w2v'].vector_size,))
        else: base = np.zeros((len(vectorizer.idf_),))

        if byW2v:
            if descs:
                people, description = map(list, zip(*[(name, np.sum(
                    [np.ravel(np.sum(model['w2v']
                        [[t for t in desc if t in model['w2v']]], axis=0).sum(axis=0)) \
                            for desc in descs[name]],
                        axis = 0
                    ) / len(descs[name])) for name in descs]))
            else:
                people, description = [], []

        else: #bm25
            if descs:
                people, description = map(list, zip(*[(name, np.sum(
                    [np.ravel(bm25_model.transform(
                        vectorizer.transform([' '.join(desc)])).sum(axis=0))
                        for desc in descs[name]] + [base], axis = 0
                    ) / max(1, len(descs[name]))) for name in descs]))
            else:
                people, description = [], []

        #pprint(list(zip(people, [np.sum(d) for d in description])))
        return people, description

    xtrain_tfv, ytrain, model['ntrain'] = map(list, zip(*
        [(desc, name in acc, name)
            for idx, acc, pred in data['TrainData']
            for _, cont in [data['corpus'][idx]]
            for name, desc in zip(*names_to_bm25(pred, cont))
            ]))


    # start
    if byW2v: svd = decomposition.TruncatedSVD()
    else: svd = decomposition.TruncatedSVD(n_components=200)

    svd.fit(xtrain_tfv)
    xtrain_svd = svd.transform(xtrain_tfv)

    scl = preprocessing.StandardScaler()
    scl.fit(xtrain_svd)
    xtrain_svd_scl = scl.transform(xtrain_svd)

    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                            subsample=0.8, nthread=10, learning_rate=0.1)
    clf.fit(xtrain_svd, ytrain)

    #def documents_to_bm25(tokens):
    #    tf = model['vectorizer'].transform(tqdm(tokens))
    #    print("doing the valid set transformation...", end='')
    #    sys.stdout.flush()
    #    DocData = model['bm25'].transform(tf)
    #    print("ok")
    #    print('tf.shape:', tf.shape)
    #    return DocData

    def validate(xvalid_tfv, show_loss = False):
        if not xvalid_tfv: return []
        xvalid_svd = svd.transform(xvalid_tfv)
        xvalid_svd_scl = scl.transform(xvalid_svd)

        if show_loss:
            predictions = clf.predict_proba(xvalid_svd)
            print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

        predictions = clf.predict(xvalid_svd)

        return predictions

    def predict(pred, cont, show_loss = False):
        name, desc = names_to_bm25(pred, cont)
        return name, validate(desc, show_loss)

    model['names_to_bm25'] = names_to_bm25
    model['validate'] = validate
    model['predict'] = predict

    return model

def extractor(amlclf=None, name = ''):

    main_name_ext = main_name_extractor()

    config = get_config('ckip', 'amlcont')
    data = load_data(config)
    #data = load_data(config, True)
    model = xgboost_model(data, amlclf)
    tokenize = token_maker('ckip')

    sub_name_ext = sub_name_extractor()

    def _ext(doc):

        beg = timeit.default_timer()
        ck_names, ws, pos, ner, time = sub_name_ext(doc)
        de = set('?？!！。,，;；:：、〔〕－／「」')
        ws = ' '.join([w for w in ws if w not in de])
        ckip_cost = timeit.default_timer() - beg
        print('sub name extractor cost: {}s, candidates: {}'.format(ckip_cost, ck_names))

        beg = timeit.default_timer()
        bt_names = main_name_ext(doc)
        bert_cost = timeit.default_timer() - beg
        print('main name extractor cost: {}s, candidates: {}'.format(bert_cost, bt_names))

        bt_names_nrpt = [name for name in bt_names if not isReporter(name, ws)]
        ck_names_nrpt = [name for name in ck_names if not isReporter(name, ws)]

        bt_names_flt, ck_names_flt = [], []
        #bt_uniq_names = list(set([n for n in bt_names_nrpt if len(n) == 1]))
        #print('bt uniq', bt_uniq_names)
        if bt_names_nrpt: bt_names_flt = model['namefilter'](bt_names_nrpt, doc)
        if ck_names_nrpt: ck_names_flt = model['namefilter'](ck_names_nrpt, doc)


        if not bt_names_flt: bt_names_flt = bt_names_nrpt
        if not ck_names_flt: ck_names_flt = ck_names_nrpt

        #recv_uniq_names = [n for n in ck_names_flt if sum([e in n for e in bt_uniq_names]) >= 2]
        #if recv_uniq_names:
        #    print('recovert uniq names:', recv_uniq_names)
        #    bt_names_flt += recv_uniq_names
        #    print('new bert candidates', bt_names_flt)


        bt_names_fin, ck_names_fin = [], []

        # xgb is not useful for bert QQ
        #if bt_names_flt: bt_names_fin = [name for name, v in zip(*model['predict'](bt_names_flt, ws)) if v]
        if bt_names_flt: bt_names_fin = bt_names_flt
        if not bt_names_fin: ck_names_fin = [name for name, v in zip(*model['predict'](ck_names_flt, ws)) if v]

        result = []
        for candidate in [bt_names_fin, ck_names_fin, bt_names_flt, ck_names_flt, \
                bt_names_nrpt, ck_names_nrpt, bt_names]:
            if candidate:
                result = candidate
                break

        result = [','.join(r.split()) if all_english(r) and len(r.split()) == 2 else r \
                     for r in result if not r.startswith('APP')]

        return result

    return _ext

if __name__ == '__main__':

    #from XgbClassifier import classifier as xgb_classifier

    #clf, amlclf = xgb_classifier()

    config = get_config('ckip', 'amlcont')

    data = load_data(config, True)

    model = xgboost_model(data, amlclf = None)

    '''
    predictions = model['predict'](['楚瑞芳', '王光遠', '華固', '錢利忠'],
        """曾 週刊 爆料 涉嫌 假冒 華固 建設 高層 涉 土地 投資 詐騙 糾紛 前 國防部 政治 作戰 總隊 總隊長 王光遠 將軍 妻 楚瑞芳 去年 華固 建設 發 傳單 自稱 遭 華固 詐騙 華固 提告 台北 地檢署 認為 楚瑞芳 涉 加重 誹謗 罪證 明確 起訴 示意圖 記者 錢利忠 台北 報導 曾 週刊 爆料 涉嫌 假冒 華固 建設 高層 涉 土地 投資 詐騙 糾紛 前 國防部 政治 作戰 總隊 總隊長 王光遠 將軍 妻 楚瑞芳 去年 華固 建設 發 傳單 自稱 遭 華固 詐騙 華固 提告 台北 地檢署 認為 楚瑞芳 涉 加重 誹謗 罪證 明確 起訴鏡 週刊 曾 今年 月 報導 指 楚瑞芳 名 彭姓 男子 自稱 華固 建設 高層 誆稱 出任 下 華固 董事長 想 整合 台北市 和平東路 樂業路口 周邊 土地 吸引 超過人 掏錢 投資 吸 金 逾 億 元 被害人 包括 林姓 房地產商 年 投資 元 請 繼續 下 閱讀 週刊 報導 提及 楚瑞芳 取信 林姓 房地產商 承諾書 親筆 簽章 指紋 內容 提及 楚瑞芳 年 月 承蒙 林男 次 義助 奧援 多萬 元 使 事業 得以 順利 完成 承諾 年 月 日 出帳 答謝 歸還 多萬 元 外 還願 贈與 億 元 北市 松江路 華固 松疆 房子戶 北市 羅斯福 路段 華固 新天地戶 當作 酬謝 投資人 詢問 華固 建設 華固 澄清 絕無 楚瑞芳 述 情節 楚瑞芳 外 詐騙 知情 強調 華固 董事長 根本 不 認識 楚瑞芳 控告 楚瑞芳 加重 誹謗 楚瑞芳 去年 間 跑到 華固 建設 樓下 路人 發 傳單 稱 遭 華固 詐騙 再 吃上 加重 誹謗 官司 不用 抽 不用 搶 現在 新聞 保證 天天 中獎 點 下載 活動 辦法 將軍 夫人 涉假冒 華固 高層 吸 金 發 傳單 稱 遭 騙 起訴"""
            )

    pprint(predictions)
    '''

    xvalid_tfv, yvalid, nvalid = map(list, zip(*
        [(desc, name in acc, name)
            for idx, acc, pred in data['ValidData']
            for _, cont in [data['corpus'][idx]]
            for name, desc in zip(*model['names_to_bm25'](pred, cont))
            ]))

    predictions = model['validate'](xvalid_tfv)

    true = (sum(1 for y in yvalid if y == 1))
    false = (sum(1 for y in yvalid if y == 0))
    true_pos = sum(1 for x, y in zip(predictions, yvalid) if x == 1 and y == 1)
    true_neg = sum(1 for x, y in zip(predictions, yvalid) if x == 0 and y == 1)
    false_pos = sum(1 for x, y in zip(predictions, yvalid) if x == 1 and y == 0)
    false_neg = sum(1 for x, y in zip(predictions, yvalid) if x == 0 and y == 0)
    print("""
tp {} / {} ({})
tn {} / {} ({})
fp {} / {} ({})
fn {} / {} ({})
hit rate = {} / {} = {}
split ability = {} / {} = {}
""".format(
    true_pos, true, true_pos / true,
    true_neg, true,  true_neg / true,
    false_pos, false, false_pos / false,
    false_neg, false, false_neg / false,
    true_pos, true, true_pos / (true),
    true_pos + false_neg, true + false, (true_pos + false_neg) / (true + false)
    ))

    if True:
        print('train miss', len(model['data']['TrainMiss']), model['data']['TrainMiss'])
        print('valid miss', len(model['data']['ValidMiss']), model['data']['ValidMiss'])
        ntrain = model['ntrain']
        for y, p, n, x in zip(yvalid, predictions, nvalid, xvalid_tfv):

            if y == 0 and p == 1:
                if n in model['reporter']:
                    print('是記者', end='')
                print(y, p, n, sum(x))

        for y, p, n, x in zip(yvalid, predictions, nvalid, xvalid_tfv):
            if y == 1 and p == 0:
                if n in model['reporter']:
                    print('是記者', end='')
                print(y, p, n, sum(x))
        print(model['reporter'])
