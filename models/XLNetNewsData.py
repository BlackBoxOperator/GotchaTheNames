from torch.utils.data import Dataset
import pandas as pd
import torch

class NewsDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.df = pd.read_csv(path)
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    def __getitem__(self, idx):
        i, title, content, label = self.df.iloc[idx, :].values
        # 將 label 文字也轉換成索引方便轉換成 tensor
        label_tensor = torch.tensor(label)
        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = []
        try:
            tokens_a = self.tokenizer.tokenize(content)
        except:
            tokens_a = self.tokenizer.tokenize("")
        word_pieces += tokens_a[-510:]+["<cls>"]  # + ["[SEP]"]
        len_a = len(word_pieces)
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a,
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
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
        self.doc = doc

    def __getitem__(self, idx):
        i, title, content, label = 0, '', self.doc, 0
        # 將 label 文字也轉換成索引方便轉換成 tensor
        label_tensor = torch.tensor(label)
        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = []
        try:
            tokens_a = self.tokenizer.tokenize(content)
        except:
            tokens_a = self.tokenizer.tokenize("")
        word_pieces += tokens_a[-510:]+["<cls>"]  # + ["[SEP]"]
        len_a = len(word_pieces)
        '''
        # 第二個句子的 BERT tokens
        tokens_b = self.tokenizer.tokenize("洗錢")
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a
        '''
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a, dtype=torch.long)
        return (tokens_tensor, segments_tensor, label_tensor)

    def __len__(self):
        return self.len


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import os
    root = os.path.join('..', 'data')
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-mid")
    trainset = NewsDataset(root+"/TrainClassify.csv", tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=True,collate_fn=create_mini_batch)
    for token, segment, mask, label in dataloader:
        a = (token,segment,mask,label)
