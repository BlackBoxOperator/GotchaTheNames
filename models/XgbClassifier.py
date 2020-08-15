#!/usr/bin/env python3.6
# coding: utf-8

from tqdm import *
import numpy as np
import time, os, json, csv, re, sys
import shutil

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
from fit import fitness

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

title_weight = 1

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


def load_data(config, split = False):

    data = {'config': config}

    token = mapTrim(open(config['tokenFile'], encoding="UTF-8").read().split('\n'))
    title = mapTrim(open(config['titleFile'], encoding="UTF-8").read().split('\n'))
    tokey = mapTrim(open(config['tokeyFile'], encoding="UTF-8").read().split('\n'))

    if len(tokey) != len(token) or len(token) != len(title):
        print('''
        input file length mismatch:
        tokey: {}
        title: {}
        token: {}'''.format(len(tokey), len(title), len(token)))
        exit(0)

    aml = [id for id, *_ in list(csv.reader(open(amlFile, encoding="UTF-8")))[1:]]
    label = np.array([int(key in aml) for key in tokey])

    if split:
        xtrain, xvalid, data['TrainLabel'], data['ValidLabel'] = \
            train_test_split(list(zip(tokey, title, token)), label,
                    test_size=0.15, stratify=label, random_state=42)
        TrainTokey, TrainTitle, TrainToken = map(list, zip(*xtrain))
        ValidTokey, ValidTitle, ValidToken = map(list, zip(*xvalid))
    else:
        TrainTokey, TrainTitle, TrainToken = tokey, title, token
        ValidTokey, ValidTitle, ValidToken = [], [], []
        data['TrainLabel'], data['ValidLabel'] = label, []

    ##append title to doc
    #print("\nappending title to document...\n")
    #for i, key in enumerate(tqdm(TrainTokey)):
    #    if TrainTitle[i] and TrainTitle[i] != "Non":
    #        TrainToken[i] += ' {}'.format(TrainTitle[i]) * title_weight
    #for i, key in enumerate(tqdm(ValidTokey)):
    #    if ValidTitle[i] and ValidTitle[i] != "Non":
    #        ValidToken[i] += ' {}'.format(ValidTitle[i]) * title_weight

    data['TrainData'], data['ValidData'] = TrainToken, ValidToken

    return data

def get_config(name, content = ''):
    config = {}

    if content: content += '_'

    config['tokenize'] = token_maker(name)
    config['tokenFile'] = os.path.join('..', 'token', '{}_{}token.txt'.format(name, content))
    config['tokeyFile'] = os.path.join('..', 'token', '{}_{}tokey.txt'.format(name, content))
    config['titleFile'] = os.path.join('..', 'token', '{}_{}title.txt'.format(name, content))

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

def xgboost_model(data):

    model = {'data' : data}

    print("\nbuilding corpus vector space...\n")

    model['bm25'] = BM25Transformer()
    model['vectorizer'] = TfidfVectorizer()
    model['vectorizer'].fit(data['TrainData'])
    #data['vectorizer'].fit(ValidToken)

    TrainTf = model['vectorizer'].transform(tqdm(data['TrainData']))

    print("fitting bm25...", end='');
    sys.stdout.flush()
    model['bm25'].fit(TrainTf)
    #data['bm25'].fit(ValidTf)
    print("ok")

    print("transforming...", end='');
    sys.stdout.flush()
    data['TrainData'] = model['bm25'].transform(TrainTf)
    print("ok")
    print('TrainTf.shape:', TrainTf.shape)

    ytrain = data['TrainLabel']
    xtrain_tfv = data['TrainData']

    svd = decomposition.TruncatedSVD(n_components=120)

    svd.fit(xtrain_tfv)
    xtrain_svd = svd.transform(xtrain_tfv)

    scl = preprocessing.StandardScaler()
    scl.fit(xtrain_svd)
    xtrain_svd_scl = scl.transform(xtrain_svd)

    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                            subsample=0.8, nthread=10, learning_rate=0.1)
    clf.fit(xtrain_svd, ytrain)

    def documents_to_bm25(tokens):
        tf = model['vectorizer'].transform(tqdm(tokens))
        print("doing the valid set transformation...", end='')
        sys.stdout.flush()
        DocData = model['bm25'].transform(tf)
        print("ok")
        print('ValidTf.shape:', tf.shape)
        return DocData

    def validate(documents, show_loss = False):
        xvalid_tfv = documents_to_bm25(documents)
        xvalid_svd = svd.transform(xvalid_tfv)
        xvalid_svd_scl = scl.transform(xvalid_svd)

        if show_loss:
            predictions = clf.predict_proba(xvalid_svd)
            print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

        predictions = clf.predict(xvalid_svd)
        return predictions

    def predict(doc, show_loss = False):
        return validate([data['config']['tokenize'](doc)], show_loss)[0]

    model['validate'] = validate
    model['predict'] = predict

    return model

def classifier(name = 'search_big'):
    data = load_data(get_config(name))
    #data = load_data(get_config(name), split=True)
    model = xgboost_model(data)
    return model['predict'], model

if __name__ == '__main__':

    data = load_data(get_config('search_big'), split=True)

    model = xgboost_model(data)

    doc = '''
    社會中心／綜合報導 房仲名師王派宏驚傳吸金25億「失蹤了」，消息一出震撼房產投資界。67年次的 王派宏自稱炒房專家，在全台授課分享理財，不過卻有顧問公司透露，上月28日開始就聯繫不上他，以為發生危險才立刻報案，但王卻已離開台灣。有學員近日出面控訴稱，4月向王派宏大量簽約後，反而血本無歸，如 今王人間蒸發，「真的會活不下去」。  ▲房仲圈驚爆王派宏「消失了」。（圖／翻攝自YouTube／王派宏） 有網友上月30日在批踢踢實業坊爆料稱，房地產老師王派宏「消失了」，文章一出引起不少人譁然。曾與王派 宏一起撰寫《學校老師沒教的賺錢秘密》一書的作者林茂盛，也在臉書發表長文提到王派宏「消失了」一事。 王派宏自稱炒房專家，在全台授課分享理財。曝光的文宣資料寫道，「很多人賺錢靠時間，有錢人賺錢靠方 法，很多資訊及檯面下的技巧，不是你不會，只是你不知道。然而這些方法和技巧實際運用，卻可以幫助你發揮十倍功力、減少十年努力！」  ▲王派宏自稱「房產幽默大師」。（圖／翻攝自YouTube／王派宏） 「賺錢 秘密巡迴分享會」宣傳資料還說，「專攻房地產與賺錢秘密多年的王派宏老師，他曾寫了一本狂銷26刷的暢銷書房地產賺錢筆記，他開辦的房地產投資課程至今已累計10000多位學員，形成派宏老師獨特的房產、賺錢秘 密資訊網，也幫助了50-60位學員達到財富自由…他手上累積了400多個實證有效的賺錢秘密，就要在講座告訴你。」  ▲王派宏「賺錢秘密分享會」的文宣訊息。（圖／翻攝自臉書投資客俱樂部粉絲專頁） 現年41歲的王 派宏在YouTube上成立官方頻道，靠著拍宣傳影片教民眾如何投資房地產。除了在網路分享炒房術、如何理財外，他甚至還開班授課，分享自身的理財投資經驗。不過近日有顧問公司透露，上月28日開始就聯繫不上王派 宏，以為發生危險才立刻報案，但王卻已離開台灣。 據了解，王派宏的投資項目非常多，除了房產、股票外，他還曾鼓吹會員「將黃金磨成粉」，進印度轉賣。王派宏的好友林茂盛4月30日在臉書發文也提到此事，「 堅守了這麼多年的原則貞操就這樣破戒了，在一番詢問很多學員去印度的狀況還有在印度工作的學員工作狀況，還查了香港的狀況，我查證了很多後就這麼傻傻的相信了。」  ▲王派宏在YouTube談「斡旋金」的秘密。 （圖／翻攝自YouTube／王派宏） 一名曾上過相關課程的王小姐4年前接受《東森新聞》訪問時表示，「王派宏都會用超便宜的費用讓大家來聽課、要你加入將近4萬塊的課程，但這還只是開始，進去才知道根本不是這 麼一回事。」 王小姐透露，王派宏當時稱「你給我30萬啊多少萬、去開那個權證代操戶給我，我保證你們賺錢，我還會開本票給你。但王小姐事後回想，「這個東西已經不是很合法了」。 另有學員近日出面控訴稱， 他在今年4月向王派宏大量簽約後，反而血本無歸，如今王人間蒸發，「真的會活不下去」。據了解，房產、投資界目前盛傳王派宏跟太太2人疑似出境到法國，受害者至少損失25億元；《ETtoday》新聞雲記者晚間實際 去電，不過截至發稿時間為止，未獲得任何回應。 ▶影／房仲圈驚傳「王派宏老師失蹤了」 曾開課曝賺錢秘密一期要4萬▶繳錢學投資最容易上當\u3000破解房產達人詐騙術 關鍵字：炒房名師﹑王派宏﹑吸金﹑黃金﹑ 印度﹑投資﹑炒股﹑25億 ※本文版權所有，非經授權，不得轉載。[ETtoday著作權聲明]※ 炒房名師王派宏疑「捲25億」失蹤了\u3000受害者嘆：真的活不下去
    '''

    print('predict result:', model['predict'](doc))

    predictions = model['validate'](data['ValidData'])
    yvalid = data['ValidLabel']

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
fn {} / {} ({})""".format(
    true_pos, true, true_pos / true,
    true_neg, true,  true_neg / true,
    false_pos, false, false_pos / false,
    false_neg, false, false_neg / false
    ))
