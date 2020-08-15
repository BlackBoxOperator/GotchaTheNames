#from XgbClassifier import classifier as xgb_classifier
from XgbExtractor import extractor as xgb_extractor
#from BertClassifier import classifier as bert_classifier
from RobertaClassifier import classifier as bert_classifier
#from BertExtractor import extractor as bert_extractor
import timeit, sys

def aml_name_extractor(name = ''):
    clf, amlclf = bert_classifier()
    #bext = bert_extractor()
    ext = xgb_extractor()
    #ext = extractor(amlclf = amlclf)
    def _extractor(doc):
        beg = timeit.default_timer()
        isAML = clf(doc)
        classifier_cost = timeit.default_timer() - beg
        print('classifier cost: {}s, isAML = {}'.format(classifier_cost, isAML))
        beg = timeit.default_timer()
        if isAML:
            names = ext(doc)
        else:
            names = []
        extractor_cost = timeit.default_timer() - beg
        print('extractor cost: {}s'.format(extractor_cost))
        print('total cost: {}s'.format(classifier_cost + extractor_cost))
        return names

    return _extractor

if __name__ == '__main__':
    from pandas import read_csv
    data = read_csv(sys.argv[1])
    ext = aml_name_extractor()
    pred = []
    for idx, cont in zip(data['index'], data['content']):
        pred.append(ext(cont))
    data['predict'] = pred
    data.to_csv(sys.argv[1] + '.out')
    #bcf, _ = bert_classifier()
    #print('init done')
    #beg = timeit.default_timer()
    #result = bcf(doc)
    #classifier_cost = timeit.default_timer() - beg
    #print('classifier cost: {}s'.format(classifier_cost))
    #print(result)
