# GotchaTheNames

Usage
---

Prepare the data (but you need prepare your training data by yourself)

```
make data   # to get ckip and dict, stopword
```

The main repo for AML news NLP competition

```
pip install requirements.txt # install packages
make server # to start the server (you need train your model first)
make query  # send naive queries to the server
```

Training

```
# note the train variable neeed to be True
python RobertaClassifier.py
python BertExtractor.py
```

Overview
---

```
.
├── cache
│   └── README.md
├── ckip
│   ├── embedding_character
│   │   ├── token_list.npy
│   │   └── vector_list.npy
│   ├── embedding_word
│   │   ├── token_list.npy
│   │   └── vector_list.npy
│   ├── LICENSE
│   ├── model_ner
│   │   ├── label_list.txt
│   │   ├── model_ontochinese_Att-0_BiLSTM-2-500_batch128-run1.data-00000-of-00001
│   │   ├── model_ontochinese_Att-0_BiLSTM-2-500_batch128-run1.index
│   │   ├── model_ontochinese_Att-0_BiLSTM-2-500_batch128-run1.meta
│   │   └── pos_list.txt
│   ├── model_pos
│   │   ├── label_list.txt
│   │   ├── model_asbc_Att-0_BiLSTM-2-500_batch256-run1.data-00000-of-00001
│   │   ├── model_asbc_Att-0_BiLSTM-2-500_batch256-run1.index
│   │   └── model_asbc_Att-0_BiLSTM-2-500_batch256-run1.meta
│   ├── model_ws
│   │   ├── model_asbc_Att-0_BiLSTM-cross-2-500_batch128-run1.data-00000-of-00001
│   │   ├── model_asbc_Att-0_BiLSTM-cross-2-500_batch128-run1.index
│   │   └── model_asbc_Att-0_BiLSTM-cross-2-500_batch128-run1.meta
│   └── README.md
├── data
│   ├── BertClassTrainValid.py
│   ├── BertExtractTrainValid.py
│   ├── dict
│   │   ├── dict.txt
│   │   ├── dict.txt.big
│   │   └── README.md
│   ├── familyName.txt
│   ├── stopword
│   │   ├── README.md
│   │   ├── simple_stopword.txt
│   │   └── stopword.txt
│   └── XLNetExtractTrainValid.py
├── fdrive.sh
├── Makefile
├── models
│   ├── aml.py
│   ├── BertClassifier.py
│   ├── BertExtractor.py
│   ├── bm25.py
│   ├── ckip_util.py
│   ├── evo.py
│   ├── fit.py
│   ├── log
│   │   └── README.md
│   ├── logger.py
│   ├── make_token.py
│   ├── NameData.py
│   ├── namefilter.py
│   ├── NewsData.py
│   ├── oracle.py
│   ├── README.md
│   ├── RobertaClassifier.py
│   ├── RobertaExtractor.py
│   ├── server.py
│   ├── W2vTrain.py
│   ├── XgbClassifier.py
│   ├── XgbExtractor.py
│   ├── XLNetClassifier.py
│   ├── XLNetExtractor.py
│   ├── XLNetNameData.py
│   └── XLNetNewsData.py
├── query
│   └── query.csv
├── README.md
├── requirements.txt
├── scripts
│   ├── amlweight
│   │   ├── extract_word.py
│   │   ├── logger.py
│   │   └── sim_words.py
│   ├── crawler
│   │   ├── back.py
│   │   ├── check_data.py
│   │   ├── crawler.py
│   │   └── extract_domain.py
│   ├── filter.py
│   └── keyword
│       ├── key_lda.txt
│       ├── key_textrank.txt
│       ├── key_tfidf.txt
│       ├── lda.py
│       ├── textrank.py
│       ├── tfidf.py
│       └── trk.py
├── test.py
├── token
│   └── README.md
└── w2v
    └── README.md

19 directories, 77 files
```
