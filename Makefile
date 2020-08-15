.PHONY: query server
PYTHON=python3.6

CKIP_PATH=ckip/ckip.rar
CKIP_FILE=12X5KASyFioNFJSGtpq_R56qq9iPzRmRs

SWORDS=data/stopword/simple_stopword.txt data/stopword/stopword.txt
SW_PATH=data/stopword/stopwords.rar
SW_FILE=1XZJIfIERTTmHobj71K8AoZXODGNyEO78

DICTS=data/dict/dict.txt data/dict/simple_dict.txt data/dict/simple_wdict.txt data/dict/wdict.txt
DC_PATH=data/dict/dicts.rar
DC_FILE=1wtHA6RMqe10gDpmW04aIcMzjIkn88uxM

$(CKIP_PATH):
	./fdrive.sh $(CKIP_FILE) $(CKIP_PATH)
	rar x $(CKIP_PATH) `dirname $(CKIP_PATH)`

$(SWORDS): | $(SW_PATH)
	rar x $(SW_PATH) `dirname $(SW_PATH)`

$(SW_PATH):
	./fdrive.sh $(SW_FILE) $(SW_PATH)

$(DICTS): | $(DC_PATH)
	rar x $(DC_PATH) `dirname $(DC_PATH)`

$(DC_PATH):
	./fdrive.sh $(DC_FILE) $(DC_PATH)

data: $(SWORDS) $(DICTS) $(CKIP_PATH)

server:
	cd models && $(PYTHON) server.py

query:
	$(PYTHON) test.py

