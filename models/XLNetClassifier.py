import os
from XLNetNewsData import create_mini_batch, NewsDataset
import torch
from transformers import XLNetForSequenceClassification,AutoModelForSequenceClassification
from transformers import XLNetTokenizer,AutoTokenizer, AdamW

root = os.path.join('..', 'data')
EPOCHS = 6
NUM_LABELS = 2
classifier_model_idx = 2
## cuda0->set1
## cuda1->set2
## cuda2->set3
CUDA_NUM = "cuda:2"
train = True
setup = "Setup1"
#setup = ""
TrainFile = "TrainClassify" + setup + ".csv"
ValidFile = "ValidClassify" + setup + ".csv"
#ValidFile = "CheckClassify.csv"
#ValidFile = "WeakClassify.csv"

PRETRAINED_MODEL_NAME = "hfl/chinese-xlnet-base"
#PRETRAINED_MODEL_NAME = "hfl/chinese-xlnet-mid"
#PRETRAINED_MODEL_NAME = "clue/xlnet_chinese_large"

if PRETRAINED_MODEL_NAME == "hfl/chinese-xlnet-base":
    ModelName = "XLNetBaseClassifierModel{}"
    BATCH_SIZE = 8
elif PRETRAINED_MODEL_NAME == "hfl/chinese-xlnet-mid":
    #BATCH_SIZE = 3
    #ModelName = "XLNetMidClassifierModel{}"
    BATCH_SIZE = 3
    ModelName = "XLNetMidB2ClassifierModel{}"
elif PRETRAINED_MODEL_NAME == "clue/xlnet_chinese_large":
    ModelName = "XLNetLrgClassifierModel{}"
    BATCH_SIZE = 1
else:
    print("wrong pretrain model name")
    exit(1)
ModelName+=setup


def get_predictions(model, dataloader,device):
    predictions = None
    correct = 0
    tp = 0
    relative = 0
    unrelative = 0
    tn = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            if next(model.parameters()).is_cuda:
                data = [t.to(device) for t in data if t is not None]

            tokens_tensors, segments_tensors, masks_tensors, labels = data
            outputs = model(input_ids=tokens_tensors,
                            # token_type_ids=segments_tensors,
                            attention_mask=masks_tensors)
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()
            for i,la in enumerate(labels):
                if la == 1:
                    relative+=1
                    if pred[i] == 1:
                        tp+=1
                else:
                    unrelative+=1
                    if pred[i]==0:
                        tn+=1

            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

    acc = correct / total
    print("relative accuracy = {}/{} = {}".format(tp,relative,tp/relative))
    print("unrelative accuracy = {}/{} = {}".format(tn,unrelative,tn/unrelative))
    return predictions, acc

def classifier():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    device = torch.device(CUDA_NUM if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, config={"num_labels":NUM_LABELS})

    model = model.to(device)
    model.load_state_dict(
            torch.load(os.path.join(ModelName.format(classifier_model_idx),
                'pytorch_model.bin'), map_location="cpu"))

    model.eval()

    def predict(doc):

        predset = NewsDoc(tokenizer=tokenizer, doc=doc)
        dataloader = torch.utils.data.DataLoader(
                predset,batch_size=1,shuffle=True,collate_fn=create_mini_batch)

        with torch.no_grad():

            data = next(iter(dataloader))

            if next(model.parameters()).is_cuda:
                data = [t.to(device) for t in data if t is not None]

            #if next(model.parameters()).is_cuda:
            #    data = [t.to(device) for t in data if t is not None]

            tokens_tensors, segments_tensors, masks_tensors, labels = data
            outputs = model(input_ids=tokens_tensors,
                            # token_type_ids=segments_tensors,
                            attention_mask=masks_tensors)
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)

        return bool(pred[0])

    return predict, None


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    device = torch.device(CUDA_NUM if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, config={"num_labels":NUM_LABELS})

    model = model.to(device)
    if train:
        trainset = NewsDataset(os.path.join(root, TrainFile), tokenizer=tokenizer)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                            collate_fn=create_mini_batch,shuffle=True)
        model.train()
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                                        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

        for epoch in range(EPOCHS):

            running_loss = 0.0
            for data in trainloader:

                tokens_tensors, segments_tensors, \
                masks_tensors, labels = [t.to(device) for t in data]

                optimizer.zero_grad()

                outputs = model(input_ids=tokens_tensors,
                                # token_type_ids=segments_tensors,
                                attention_mask=masks_tensors,
                                labels=labels)

                loss = outputs[0]
                loss.backward()
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
            trainset = NewsDataset(os.path.join(root, ValidFile), tokenizer=tokenizer)
            testloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                collate_fn=create_mini_batch)
            _, acc = get_predictions(model, testloader, device=device)
            print(acc)
