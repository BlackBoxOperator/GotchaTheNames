from torch.utils.data import Dataset
import pandas as pd
import torch

class NewsDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.df = pd.read_csv(path)
        self.len = len(self.df)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        i, title, content, label = self.df.iloc[idx, :].values
        label_tensor = torch.tensor(label)
        word_pieces = ["[CLS]"]
        try:
            tokens_a = self.tokenizer.tokenize(content)
        except:
            tokens_a = self.tokenizer.tokenize("")
        word_pieces += tokens_a[-510:]  # + ["[SEP]"]
        len_a = len(word_pieces)
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        segments_tensor = torch.tensor([1] * len_a,
                                        dtype=torch.long)
        return (tokens_tensor, segments_tensor, label_tensor)

    def __len__(self):
        return self.len

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None

    tokens_tensors = torch.nn.utils.rnn.pad_sequence(tokens_tensors,
                                  batch_first=True)
    segments_tensors = torch.nn.utils.rnn.pad_sequence(segments_tensors,
                                    batch_first=True)

    masks_tensors = torch.zeros(tokens_tensors.shape,
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, masks_tensors, label_ids

class NewsDoc(Dataset):
    def __init__(self, tokenizer, doc):
        self.len = 1
        self.tokenizer = tokenizer
        self.doc = doc

    def __getitem__(self, idx):
        i, title, content, label = 0, '', self.doc, 0
        label_tensor = torch.tensor(label)
        word_pieces = ["[CLS]"]
        try:
            tokens_a = self.tokenizer.tokenize(content)
        except:
            tokens_a = self.tokenizer.tokenize("")
        word_pieces += tokens_a[-510:]
        len_a = len(word_pieces)
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        segments_tensor = torch.tensor([1] * len_a, dtype=torch.long)
        return (tokens_tensor, segments_tensor, label_tensor)

    def __len__(self):
        return self.len


if __name__ == "__main__":
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    trainset = NewsDataset("TrainClassify.csv", tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=True,collate_fn=create_mini_batch)
    for token, segment, mask, label in dataloader:
        a = (token,segment,mask,label)
