from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import re
import textwrap
import unicodedata

"""
tag2idx={
# 'B-art': 14,
# 'B-eve': 16,
# 'B-geo': 0,
# 'B-gpe': 13,
# 'B-nat': 12,
# 'B-org': 10,
 'B-per': 4,
# 'B-tim': 2,
# 'I-art': 5,
# 'I-eve': 7,
# 'I-geo': 15,
# 'I-gpe': 8,
# 'I-nat': 11,
# 'I-org': 3,
 'I-per': 6,
# 'I-tim': 1,
 'X'    :17,
 'O'    : 9,
 '<cls>':18,
 '<sep>':19
}
"""

tag2idx={
# 'B-art': 14,
# 'B-eve': 16,
# 'B-geo': 0,
# 'B-gpe': 13,
# 'B-nat': 12,
# 'B-org': 10,
 'B-per': 0,
# 'B-tim': 2,
# 'I-art': 5,
# 'I-eve': 7,
# 'I-geo': 15,
# 'I-gpe': 8,
# 'I-nat': 11,
# 'I-org': 3,
 'I-per': 1,
# 'I-tim': 1,
 'X'    : 2,
 'O'    : 3,
 '<cls>': 4,
 '<sep>': 5
}

tag2name = { tag2idx[k] : k for k in tag2idx }

def isMsk(t):
    return t[0] == '[' and t[-1] == ']'

def recovery_token(string, token):
    orig = token
    ostr = string
    token = [t for t in token]
    string = unicodedata.normalize('NFKC', string)
    recov, unknown = [], 0
    while token:
        t = token[0]
        if len(t) > 1 and t.startswith('▁'): t = t[1:] 

        if t in ('<cls>', '<sep>', '▁'):
            recov.append('')
        elif t.strip() and isMsk(t):
            unknown += 1
        elif t in string and unknown:
            p = string.index(t)
            d = p // unknown
            if not d: break
            recov += textwrap.wrap(string[:p], d)
            string = string[p:].strip()
            unknown = 0
            continue
        elif string.startswith(t):
            recov.append(t)
            string = string[len(t):].strip()
        else:
            break
        token.pop(0)

    if token:
        print('original string', ostr)
        print('original tokens', orig)
        print(token)
        print(recov)
        print('recover failed, "{}", not match "{}"'.format(t, string))
        return orig
    
    return recov

class NameDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.df = pd.read_csv(path)
        self.len = len(self.df)
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    def __getitem__(self, idx):
        _, token, label = self.df.iloc[idx, :].values
        # 將 label 文字也轉換成索引方便轉換成 tensor

        #label = np.array([tag2idx[l] for l in (eval(label)[:510] + ["<sep>"] + ["<cls>"])])
        label = np.array([tag2idx[l] for l in (eval(label)[:510] + ["<cls>"])])
        label_tensor = torch.tensor(label)

        # 建立第一個句子的 BERT tokens 並加入分隔符號 <sep>
        #word_pieces = eval(token)[:510] + ["<sep>"] + ["<cls>"]
        word_pieces = eval(token)[:510] + ["<cls>"]
        
        len_a = len(word_pieces)
        
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        # Seem like for sequance tagging task, it's not necessary to make this embedding
        segments_tensor = torch.tensor([0] * len_a, dtype=torch.long)

        return (tokens_tensor, segments_tensor, label_tensor, word_pieces)

    def __len__(self):
        return self.len

def create_mini_batch(samples):

    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    label_tensors = [s[2] for s in samples]
    orig_tokens = [s[3] for s in samples]

    tokens_tensors = torch.nn.utils.rnn.pad_sequence(
            tokens_tensors, batch_first=True)

    segments_tensors = torch.nn.utils.rnn.pad_sequence(
            segments_tensors, batch_first=True)

    label_tensors = torch.nn.utils.rnn.pad_sequence(
            label_tensors, batch_first=True, padding_value=tag2idx["O"])

    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors > 0, 1)

    return orig_tokens, tokens_tensors, segments_tensors, masks_tensors, label_tensors

class NameDoc(Dataset):
    def __init__(self, tokenizer, doc):

        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

        toks = []
        docs = []
        for sent in [c.strip() for c in re.split('。', doc.replace('。」', '」').replace('。）', '）'))]:
            sent += '。'
            tok = tokenizer.tokenize(sent)
            if len(tok) < 8:
                continue
            toks.append(tok)
            docs.append(sent)

        self.toks = toks
        self.docs = docs
        self.len = len(docs)

    def __getitem__(self, idx):
        doc, token, label = self.docs[idx], self.toks[idx], None
        # 建立第一個句子的 BERT tokens 並加入分隔符號 <sep>
        #word_pieces =  token[:510] + ["<sep>"]+["<cls>"]
        word_pieces =  token[:510] + ["<cls>"]
        len_a = len(word_pieces)

        # 將 label 文字也轉換成索引方便轉換成 tensor
        #label_tensor = torch.tensor([0] * len(token))
        label_tensor = torch.tensor([0] * len_a)

        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        # 將第一句包含 <sep> 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([1] * len_a, dtype=torch.long)

        return (tokens_tensor, segments_tensor, label_tensor, recovery_token(doc, word_pieces))

    def __len__(self):
        return self.len


if __name__ == "__main__":
    pass
#    from transformers import BertTokenizer
#    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
#    trainset = NameDataset(TrainFile, tokenizer=tokenizer)
#    dataloader = torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=True,collate_fn=create_mini_batch)
#    for token, segment, mask, label in dataloader:
#        a = (token,segment,mask,label)
