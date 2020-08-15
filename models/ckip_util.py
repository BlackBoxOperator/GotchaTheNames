import os, re
import timeit
import numpy as np

use_gpu = 0

# hide tf message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

ckipt_data = os.path.join('..', 'ckip')
w2vPath = os.path.join(ckipt_data, 'embedding_word')
c2vPath = os.path.join(ckipt_data, 'embedding_character')

if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_tokenizer():
    from ckiptagger import WS
    ws = WS(ckipt_data, disable_cuda=not use_gpu)
    return lambda doc: ws([doc])[0]

def exist_chinese(s):
    return len(re.findall(r'[\u4e00-\u9fff]+', s))

def all_chinese(s):
    n = ''.join(s.split())
    m = re.findall(r'[\u4e00-\u9fff]+', n)
    return m and len(m[0]) == len(n)

def exist_english(s):
    return any([c.isalpha() for c in s])

def all_english(s):
    n = ''.join(s.split())
    return all([c.isalpha() for c in n])

def name_extractor():
    from ckiptagger import WS, POS, NER
    ws = WS(ckipt_data, disable_cuda=not use_gpu)
    pos = POS(ckipt_data, disable_cuda=not use_gpu)
    ner = NER(ckipt_data, disable_cuda=not use_gpu)

    def extract_name(doc, attr='PERSON'):
        start = timeit.default_timer()
        word_s = ws([doc],
                    sentence_segmentation=True,
                    segment_delimiter_set={
                        '?', '？', '!', '！', '。',
                        ',','，', ';', ':', '、'})
        word_p = pos(word_s)
        word_n = ner(word_s, word_p)
        stop = timeit.default_timer()

        namelist = set([e[3] for e in word_n[0] if e[2] == attr and len(e[3]) > 1 and '、' not in e[3] and e[3][-1] not in '案犯'])

        return namelist, word_s[0], word_p[0], word_n[0], stop - start
    return extract_name

def load_embedding(embedding_dir):
    from gensim.models.keyedvectors import Word2VecKeyedVectors
    token_file = os.path.join(embedding_dir, "token_list.npy")
    token_list = np.load(token_file)
    vector_file = os.path.join(embedding_dir, "vector_list.npy")
    vector_list = np.load(vector_file)
    model = Word2VecKeyedVectors(vector_list.shape[1])
    model.add(token_list, vector_list)
    return model.wv
    #token_to_vector = dict(zip(token_list, vector_list))
    #return {token_list[i] : vector_list[i] for i in range(d)}

def ckip_w2v():
    return load_embedding(w2vPath)

def ckip_c2v():
    return load_embedding(c2vPath)

if __name__ == '__main__':
    """
    w2v = load_embedding(w2vPath)
    c2v = load_embedding(c2vPath)
    """
    news =  '台北地院審理生產東亞燈具的上市公司中國電器掏空案，認定中電前董事長周麗真利用CLS等6家海外公司掏空資產，合議庭今天依證交法侵占罪判周有期徒刑12年，併科罰金3億元，重罪可能伴隨逃亡，合議庭宣判後加開調查庭，裁定周麗真3億元交保，併限制住居、出境、出海。 同案共犯中電公司前總經理張志偉，合議庭也依證交法侵占罪判他有期徒刑9年6月，併科罰金1億元，宣判後另裁定張志偉以1億元交保，併限制住居、出境、出海，法官對周、張2人重判後，再祭出重保等強制處分，目的是以確保日後審判進行，防止2人逃亡。 周麗真與中電公司前總經理張志偉利用成立CLS、GLI等6家海外公司掏空中電資產，並找陳逢璿擔任CLS人頭負責人，但2人實際仍掌控公司經營；2011年11月間，周麗真、張志偉指示中電藉預付款、投資款等多種名目，以445萬元美金匯給CLS，中電公司，卻未製作交易傳票，也未在中電年度財報上揭露，涉嫌假交易掏空。 此外，周麗真、張志偉明知中電百分百投資的海外子公司GLI在2011年11月間，以現金增資發行新股名義籌資750萬美金，不過，實際上GLI增資款並非由中電公司認購，而是由周、張掌握的CLS公司斥資750萬元美金認購。 檢方認定，掏空案造成原本由中電百分百投資的海外子公司GLI，中電掌握的GLI股權被稀釋至40%，GLI也因中電持股未超過50%，不必與中電合併財報，甚至張志偉還指示GLI以每股18.8元美金高價收購自己的SAWTRY公司41萬餘股股票，掏空公司資產2億3000萬元。'
    news2 = '摩根亞太集團董事長張堯勇指出，避險基金規模約為5兆美元，採取量化交易的基金規模約1兆美元，比重佔了20%，代表量化交易的操作績效好，才會有那麼高的比重。'
    extract_name = name_extractor()
    names = extract_name(news)
    print('\n'.join(names))
