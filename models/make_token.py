import os, sys, csv, re, jieba
from pprint import pprint
from tqdm import *

from gensim.models import Word2Vec

import ckip_util

csv.field_size_limit(sys.maxsize)

#w2vFile = 'news_d200_e100'

def retain_chinese(line):
    return re.compile(r"[^\u4e00-\u9fa5]").sub('',line).replace('臺', '台')

method = {
    'all': {
        'info': 'jieba all mode',
        'toker': lambda: lambda text: jieba.cut(text, cut_all = True),
        'stop': os.path.join('..', 'data', 'stopword', "stopword.txt"),
        'w2v': lambda path: lambda: Word2Vec.load(path).wv,
        'filter': retain_chinese,
    },
    'search': {
        'info': 'jieba search mode',
        'toker': lambda: lambda text: jieba.cut_for_search(text),
        'stop': os.path.join('..', 'data', 'stopword', "stopword.txt"),
        'w2v': lambda path: lambda: Word2Vec.load(path).wv,
        'filter': retain_chinese,
    },
    'search_big': {
        'info': 'jieba search mode',
        'toker': lambda: lambda text: jieba.cut_for_search(text),
        'stop': os.path.join('..', 'data', 'stopword', "stopword.txt"),
        'w2v': lambda path: lambda: Word2Vec.load(path).wv,
        'dictionary': os.path.join('..', 'data', 'dict', 'dict.txt.big'),
        'filter': retain_chinese,
    },
    'pad': {
        'info': 'jieba all mode with paddle',
        'toker': lambda: lambda text: jieba.cut(text, cut_all = True, use_paddle=True),
        'stop': os.path.join('..', 'data', 'stopword', "stopword.txt"),
        'w2v': lambda path: lambda: Word2Vec.load(path).wv,
        'dictionary': os.path.join('..', 'data', 'dict', 'dict.txt.big'),
        'filter': retain_chinese,
    },
    'ckip': {
        'info': 'ckip word segmentation',
        'toker': lambda: ckip_util.get_tokenizer(),
        #'dict': os.path.join('..', 'data', 'dict', 'dict.txt'),
        #'stop': os.path.join('..', 'data', 'stopword', "stopword.txt"),
        'w2v': lambda path: ckip_util.ckip_w2v,
        'c2v': ckip_util.ckip_c2v,
    },
    #'search_dict': {
    #    'info': 'jieba search mode with dict',
    #    'toker': lambda: lambda text: jieba.cut_for_search(text),
    #    'dict': os.path.join('..', 'data', 'dict', 'dict.txt'),
    #    'stop': os.path.join('..', 'data', 'stopword', "stopword.txt"),
    #    'w2v': lambda path: lambda: Word2Vec.load(path).wv,
    #},
    #'search_dict_bot': {
    #    'info': 'jieba search mode with dict and keyword BOT',
    #    'toker': lambda: lambda text: jieba.cut_for_search(text),
    #    'dict': os.path.join('..', 'data', 'dict', 'dict.txt'),
    #    'stop': os.path.join('..', 'data', 'stopword', "stopword.txt"),
    #    'keyword': ['BOT'],
    #    'w2v': lambda path: lambda: Word2Vec.load(path).wv,
    #},
    #'search_wdict': {
    #    'info': 'jieba search mode with wiki entry dict',
    #    'toker': lambda: lambda text: jieba.cut_for_search(text),
    #    'dict': os.path.join('..', 'data', 'dict', 'wdict.txt'),
    #    'stop': os.path.join('..', 'data', 'stopword', "stopword.txt"),
    #    'w2v': lambda path: lambda: Word2Vec.load(path).wv,
    #},
    #'search_simple_dict': {
    #    'info': 'jieba search mode with simplified dict',
    #    'toker': lambda: lambda text: jieba.cut_for_search(text),
    #    'dict': os.path.join('..', 'data', 'dict', 'simple_dict.txt'),
    #    'stop': os.path.join('..', 'data', 'stopword', "simple_stopword.txt"),
    #    'w2v': lambda path: lambda: Word2Vec.load(path).wv,
    #},
    #'search_simple_wdict': {
    #    'info': 'jieba search mode with simplified wiki dict',
    #    'toker': lambda: lambda text: jieba.cut_for_search(text),
    #    'dict': os.path.join('..', 'data', 'dict', 'simple_wdict.txt'),
    #    'stop': os.path.join('..', 'data', 'stopword', "simple_stopword.txt"),
    #    'w2v': lambda path: lambda: Word2Vec.load(path).wv,
    #},
    #'search_simple_wdict2': {
    #    'info': 'jieba search mode with simplified wiki dict ver.2',
    #    'toker': lambda: lambda text: jieba.cut_for_search(text),
    #    'dict': os.path.join('..', 'data', 'dict', 'simple_wdict2.txt'),
    #    'stop': os.path.join('..', 'data', 'stopword', "simple_stopword.txt"),
    #    'w2v': lambda path: lambda: Word2Vec.load(path).wv,
    #},
    #'search_simple_wdict3': {
    #    'info': 'jieba search mode with simplified wiki dict ver.3',
    #    'toker': lambda: lambda text: jieba.cut_for_search(text),
    #    'dict': os.path.join('..', 'data', 'dict', 'simple_wdict2.txt'),
    #    'stop': os.path.join('..', 'data', 'stopword', "simple_stopword.txt"),
    #    'w2v': lambda path: lambda: Word2Vec.load(path).wv,
    #},
    #'2gram': {
    #    'info': '2 gram',
    #    'gram': [2],
    #    'stop': os.path.join('..', 'data', 'stopword', "stopword.txt"),
    #    'w2v': lambda path: lambda: Word2Vec.load(path).wv,
    #},
    #'3gram': {
    #    'info': '3 gram',
    #    'gram': [3],
    #    'stop': os.path.join('..', 'data', 'stopword', "stopword.txt"),
    #    'w2v': lambda path: lambda: Word2Vec.load(path).wv,
    #},
}

def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)

def ngram(li, n):
    grams = []
    for s in li:
        if len(s) <= n:
            grams += [s]
            continue
        for beg in range(n):
            grams += [s[i:i+n] for i in range(beg,len(s), n)]
    return grams

def usage(cust_args = '[content.csv]'):
    print("Usage: python {} mode {}\n".format(sys.argv[0], cust_args))
    print("mode:")
    print('\n'.join(['    {:25s}: {}'.format(k, method[k]['info']) for k in method]))


def token_maker(name):

    if 'stop' in method[name]:
        stopwordTxt = method[name]['stop']

        if not os.path.isfile(stopwordTxt):
            print(stopwordTxt, 'not found.\nuse `make data` to prepare it.')
            exit(1)

        print("reading stopword file... ", end='')
        stop = open(stopwordTxt,"r")
        stopList = set(stop.read().split())
        print("done")
    else:
        stopList = set()

    if 'toker' in method[name]:
        jieba.setLogLevel(20)

    if 'dictionary' in method[name]:
        jieba.set_dictionary(method[name]['dictionary'])

    if 'dict' in method[name]:
        if not os.path.isfile(method[name]['dict']):
            print(method[name]['dict'], 'not found.\nuse `make data` to prepare it.')
            exit(1)
        jieba.load_userdict(method[name]['dict'])

    tokenize = method[name]['toker']()

    def make_token(cont):

        text = cont

        tokens = ''

        if 'toker' in method[name]:
            tokens += ' '.join([w for w in tokenize(cont)
                                    if w.strip() and w not in stopList])

        if 'gram' in method[name]:
            for n in method[name]['gram']:
                if tokens: tokens += ' '
                tokens += ' '.join([w for w in ngram(cont.split(), n)
                                        if w.strip() and w not in stopList])

        if 'keyword' in method[name]:
            raw_text = strQ2B(cont).upper()
            extra_keywords = []
            for word in method[name]['keyword']:
                word = word.upper()
                extra_keywords.append(' '.join([word] * raw_text.count(word)))

            if extra_keywords:
                if tokens: tokens += ' '
                tokens += ' '.join(extra_keywords)

        if 'filter' in method[name]:
            tokens = ' '.join([retain_chinese(t).strip() for t in tokens.split()])

        return tokens

    return make_token

if __name__ == '__main__':

    if len(sys.argv) < 2:
        usage()
        exit(0)
    else:
        name = sys.argv[1]

    if name not in method:
        print("no such mode")
        usage()
        exit(1)

    if len(sys.argv) >= 3:
        content = sys.argv[2]
        cname = os.path.splitext(os.path.basename(content))[0]
    else:
        content = os.path.join('..', 'data', 'content.csv')
        cname = ''

    from oracle import getConfig

    config = getConfig(name, check_w2v = False, check_token = False, tokname = cname)

    make_token = config['tokenize']

    tokeyFile = config['tokeyFile']
    titleFile = config['titleFile']
    tokenFile = config['tokenFile']

    if input('save tokey as {}? enter to continue:'.format(tokeyFile)).strip(): exit(1)
    if input('save title as {}? enter to continue:'.format(titleFile)).strip(): exit(1)
    if input('save token as {}? enter to continue:'.format(tokenFile)).strip(): exit(1)

    print("reading content file... ", end='')
    sys.stdout.flush()
    csvfile = open(content, 'r')
    print("done")

    mode = 'w'
    tokey = open(tokeyFile, mode)
    title = open(titleFile, mode)
    token = open(tokenFile, mode)

    csvr = csv.reader(csvfile);
    content = list(csvr)
    for idx, cap, cont, *_ in tqdm(content):
        cap = make_token(cap).strip()
        cont = make_token(cont).strip()
        cap = 'Non' if not cap else cap
        if idx and cap and cont:
            tokey.write(idx + '\n')
            title.write(cap + '\n')
            token.write(cont + '\n')

    tokey.close()
    title.close()
    token.close()
