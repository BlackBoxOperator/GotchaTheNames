import os
from NewsData import create_mini_batch, NewsDataset, NewsDoc
import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

root = os.path.join('..', 'data')
#PRETRAINED_MODEL_NAME = "clue/roberta_chinese_base"
PRETRAINED_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
#PRETRAINED_MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"
NUM_LABELS = 2
BATCH_SIZE = 2
classifier_model_idx = 4

train = True
#EPOCHS = 5
EPOCHS = 7

cuda_num = "cuda:0"
setup = "Setup1"
#setup = ""
TrainFile = "TrainClassify" + setup + ".csv"
ValidFile = "ValidClassify" + setup + ".csv"
#ValidFile = "CheckClassify.csv"
#ValidFile = "WeakClassify.csv"
ModelName = "RobertaWwmExtClassifier" + setup + "Model{}"

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
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

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
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)


    model = model.to(device)
    if train:
        trainset = NewsDataset(os.path.join(root, TrainFile), tokenizer=tokenizer)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                            collate_fn=create_mini_batch,shuffle=True)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

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
