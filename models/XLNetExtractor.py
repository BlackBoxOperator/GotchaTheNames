import os
from XLNetNameData import create_mini_batch, NameDataset, NameDoc, tag2idx, tag2name
import torch
from transformers import AutoModelForTokenClassification,AutoConfig
from transformers import AutoTokenizer, AdamW
import itertools, operator
import torch.nn.functional as F
from seqeval.metrics import classification_report,accuracy_score,f1_score
from namefilter import all_english, namefilter

root = os.path.join('..', 'data')

train = False
EPOCHS = 8
model_idx = 7
CUDA_NUM = "cuda:0"

PRETRAINED_MODEL_NAME = "hfl/chinese-xlnet-base"
#PRETRAINED_MODEL_NAME = "hfl/chinese-xlnet-mid"
#PRETRAINED_MODEL_NAME = "clue/xlnet_chinese_large"
TrainFile = "TrainXLNetExtract.csv"
ValidFile = "ValidXLNetExtract.csv"

if PRETRAINED_MODEL_NAME == "hfl/chinese-xlnet-base":
    ModelName = "XLNetBaseExtractorModel{}"
    BATCH_SIZE = 8
elif PRETRAINED_MODEL_NAME == "hfl/chinese-xlnet-mid":
    #BATCH_SIZE = 3
    #ModelName = "XLNetMidExtractorModel{}"
    BATCH_SIZE = 2
    ModelName = "XLNetMidB2ExtractorModel{}"
elif PRETRAINED_MODEL_NAME == "clue/xlnet_chinese_large":
    ModelName = "XLNetLrgExtractorModel{}"
    BATCH_SIZE = 1
else:
    print("wrong pretrain model name")
    exit(1)

def get_predictions(model, dataloader,device):

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []

    model.eval()
    predictions, hit, miss, err, total = [], 0, 0, 0, 0
    with torch.no_grad():
        for _, *data in dataloader:
            if next(model.parameters()).is_cuda:
                data = [t.to(device) for t in data if t is not None]

            tokens_tensors, segments_tensors, masks_tensors, labels = data


            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=None,
                            # token_type_ids=segments_tensors,
                            attention_mask=masks_tensors)

            logits = outputs[0]

            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()

            # Get NER true result
            labels = labels.to('cpu').numpy()


            # Only predict the real word, mark=0, will not calculate
            masks_tensors = masks_tensors.to('cpu').numpy()

            # Compare the valuable predict result
            for i,mask in enumerate(masks_tensors):
                # Real one
                temp_1 = []
                # Predict one
                temp_2 = []

                for j, m in enumerate(mask):
                    # Mark=0, meaning its a pad word, dont compare
                    if m:
                        if tag2name[labels[i][j]] != "X" \
                            and tag2name[labels[i][j]] != "<cls>" \
                            and tag2name[labels[i][j]] != "<sep>" : # Exclude the X label
                            temp_1.append(tag2name[labels[i][j]])
                            temp_2.append(tag2name[logits[i][j]])
                    else:
                        break


                y_true.append(temp_1)
                y_pred.append(temp_2)

                mtag = lambda labs: [tag2idx['I-per'] \
                        if l == tag2idx['B-per'] else l for l in labs]

                aseq = set([tuple(i for i,value in it) \
                        for key,it in itertools.groupby(
                            enumerate(mtag(labels[i])), key=operator.itemgetter(1)) \
                        if key == tag2idx['I-per']])
                pseq = set([tuple(i for i,value in it) \
                        for key,it in itertools.groupby(
                            enumerate(mtag(logits[i])), key=operator.itemgetter(1)) \
                        if key == tag2idx['I-per']])
 
                total += len(aseq)
                hit += len(pseq & aseq)
                miss += len(aseq - pseq)
                err += len(pseq - aseq)

            ##predictions.append(pseq)

    print("f1 socre: %f"%(f1_score(y_true, y_pred)))
    print("Accuracy score: %f"%(accuracy_score(y_true, y_pred)))
    print("Name score hit: {} / {} = {}".format(hit, total, hit / total))
    print("Name score miss: {} / {} = {}".format(miss, total, miss / total))
    print("Name score error: {} / {} = {}".format(err, total, err / total))
    return None, accuracy_score(y_true, y_pred)

def extractor():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    device = torch.device(CUDA_NUM if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=len(tag2idx))

    model = AutoModelForTokenClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,config=config)

    model = model.to(device)
    model.load_state_dict(
            torch.load(os.path.join(ModelName.format(model_idx),
                'pytorch_model.bin'), map_location="cpu"))

    model.eval()

    def predict(doc):

        names, docs = [], []
        predset = NameDoc(tokenizer=tokenizer, doc=doc)
        dataloader = torch.utils.data.DataLoader(
                predset,batch_size=1,shuffle=True,collate_fn=create_mini_batch)

        with torch.no_grad():

            for tokens, *data in dataloader:

                if next(model.parameters()).is_cuda:
                    data = [t.to(device) for t in data if t is not None]

                tokens_tensors, segments_tensors, masks_tensors, labels = data


                outputs = model(input_ids=tokens_tensors,
                                token_type_ids=None,
                                # token_type_ids=segments_tensors,
                                attention_mask=masks_tensors)

                logits = outputs[0]

                logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
                logits = logits.detach().cpu().numpy()

                # Only predict the real word, mark=0, will not calculate
                masks_tensors = masks_tensors.to('cpu').numpy()

                tr = lambda e: ','.join(e) if len(e) == 2 and all_english(''.join(e), '▁') else ''.join(e)

                for i,mask in enumerate(masks_tensors):
                    name, doc = [], ''
                    names.append([])
                    for j, m in enumerate(mask):
                        if m:
                            if logits[i][j] not in (tag2idx['<cls>'], tag2idx['<sep>']):
                                doc += tokens[i][j]
                            if logits[i][j] == tag2idx['B-per']:
                                if name:
                                    names[-1].append(tr(name))
                                    name = []
                                name.append(tokens[i][j])
                            elif logits[i][j] == tag2idx['I-per']:
                                name.append(tokens[i][j])
                            elif name:
                                names[-1].append(tr(name))
                                name = []
                        else:
                            break
                    if name: names[-1].append(tr(name))
                    docs.append(doc)

        # need filter the names from doc and do classification again
        return names, docs

    nft = namefilter()
    def _ext(doc):
        names, docs = predict(doc)
        print('original names', names)
        return nft(list(set().union(*names)), ''.join(docs))
        
    return _ext


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    device = torch.device(CUDA_NUM if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=len(tag2idx))

    model = AutoModelForTokenClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,config=config)

    model = model.to(device)

    # additional
    max_grad_norm = 1.0
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        # Fine tune model all layer parameters
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        # Only fine tune classifier parameters
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)


    if train:
        trainset = NameDataset(os.path.join(root, TrainFile), tokenizer=tokenizer)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                            collate_fn=create_mini_batch,shuffle=True)
        model.train()

        # use above instead
        #optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

        for epoch in range(EPOCHS):

            running_loss = 0.0
            for _, *data in trainloader:

                tokens_tensors, segments_tensors, \
                masks_tensors, labels = [t.to(device) for t in data]

                optimizer.zero_grad()

                outputs = model(input_ids=tokens_tensors,
                                token_type_ids=None,
                                # token_type_ids=segments_tensors,
                                attention_mask=masks_tensors,
                                labels=labels)

                loss = outputs[0]
                loss.backward()

                # gradient clipping, additional
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

                optimizer.step()

                running_loss += loss.item()

            _, acc = get_predictions(model, trainloader, device=device)

            print('[epoch %d] loss: %.3f, acc: %.3f' %
                (epoch + 1, running_loss, acc))
            model.save_pretrained(ModelName.format(epoch))
    else:
        for model_idx in range(0, EPOCHS):
            print("epoch {}:".format(model_idx))
            model.load_state_dict(
                    torch.load(os.path.join(ModelName.format(model_idx),
                        'pytorch_model.bin'), map_location="cpu"))
            trainset = NameDataset(os.path.join(root, ValidFile), tokenizer=tokenizer)
            testloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                collate_fn=create_mini_batch)
            _, acc = get_predictions(model, testloader, device=device)
            print(acc)
