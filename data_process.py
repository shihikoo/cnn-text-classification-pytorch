# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:37:11 2019
ref: github/GokuMohandas/practicalAI: notebooks/12_Embeddings.ipynb
@author: qwang
"""
import os
import re
import csv
import json

from argparse import Namespace
import collections
from collections import Counter

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


#%% ==================================== Set up ====================================
# Set Numpy and PyTorch seeds
def set_seeds(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        
# Creating directories
def create_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

#rob_name = 'randomisation'
#rob_name = 'blinded'
rob_name = 'ssz'
       
# Arguments
args = Namespace(
    seed=1234,
    cuda=True,
    shuffle=True,
    data_file="rob.csv",
    vectorizer_file='vectorizer.json',
    model_state_file="model.pth",
    save_dir="torchfile" + "_" + rob_name,
    train_size=0.80,
    val_size=0.10,
    test_size=0.10,
    cutoff=25, # token must appear at least <cutoff> times to be in SequenceVocabulary
    num_epochs=15,
    early_stopping_criteria=5,
    learning_rate=1e-3,
    batch_size=64,
    max_seq_len = 5000,
    num_filters=100,
    embedding_dim=200,
    hidden_dim=100,
    dropout_p=0.5,
)

# Set seeds
set_seeds(seed=args.seed, cuda=args.cuda)

# Create save dir
create_dirs(args.save_dir)

# Expand filepaths
args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
args.model_state_file = os.path.join(args.save_dir, args.model_state_file)

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")
print("Using CUDA: {}".format(args.cuda))   



#%% ==================================== Data ====================================
csv.field_size_limit(100000000)
#dat = pd.read_csv("datafile/dataWithFullText_utf8.txt", sep='\t', engine="python", encoding="utf-8")   
#dat['text'] = dat['CleanFullText']
#dat['label_random'] = dat['RandomizationTreatmentControl'] 
#dat['label_blind'] = dat['BlindedOutcomeAssessment'] 
#dat['label_ssz'] = dat['SampleSizeCalculation'] 
#dat = dat[-dat["ID"].isin([8, 608, 647, 703, 807, 903, 960, 1446, 1707, 1707, 1714, 1716, 1754, 2994, 
#                           2995, 2996, 2997, 3943, 4045, 4064, 4066, 4076, 4077, 4083, 3804, 4035])]
#dat.set_index(pd.Series(range(0, len(dat))), inplace=True)
#dat.to_csv("datafile/fulldata.csv", sep='\t', encoding='utf-8', index=False)


# Final raw data
df = pd.read_csv("datafile/fulldata.csv", usecols=['text', 'label_random', 'label_blind', 'label_ssz'], sep = '\t', engine = 'python', encoding='utf-8')
#df.loc[df.label_random==1, 'label'] = 'random'
#df.loc[df.label_random==0, 'label'] = 'non-random'
#df.loc[df.label_blind==1, 'label'] = 'blinded'
#df.loc[df.label_blind==0, 'label'] = 'non-blinded'
df.loc[df.label_ssz==1, 'label'] = 'ssz'
df.loc[df.label_ssz==0, 'label'] = 'non-ssz'
df.label.value_counts()

# Split by label
by_label = collections.defaultdict(list)
for _, row in df.iterrows():
    by_label[row.label].append(row.to_dict())
for label in by_label:
    print ("{0}: {1}".format(label, len(by_label[label])))
    
# Create split data
final_list = []
for _, item_list in sorted(by_label.items()):
    if args.shuffle:
        np.random.shuffle(item_list)
    n = len(item_list)
    n_train = int(args.train_size*n)
    n_val = int(args.val_size*n)
    n_test = int(args.test_size*n)

    # Give data point a split attribute
    for item in item_list[:n_train]:
        item['split'] = 'train'
    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'
    for item in item_list[n_train+n_val:]:
        item['split'] = 'test'  
    # Add to final list
    final_list.extend(item_list)
   
    
# df with split datasets
split_df = pd.DataFrame(final_list)
split_df["split"].value_counts()


# Preprocessing
def preprocess_text(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"[!%^&*()=_+{};:$£€@~#|/,.<>?\`\'\"\[\]\\]", " ", text)  # [!%^&*()=_+{};:$£€@~#|/<>?\`\'\"\[\]\\]
    text = re.sub(r'\b(\w{1})\b', '', text) 
    text = re.sub(r'\s+', ' ', text)
    return text.lower()
    
split_df.text = split_df.text.apply(preprocess_text)
#a=split_df.text[0]


#%% ==================================== Vocabulary ====================================
class Vocabulary(object):
    def __init__(self, token_to_idx=None):

        # Token to index
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx = token_to_idx

        # Index to token
        self.idx_to_token = {idx: token \
                             for token, idx in self.token_to_idx.items()}

    def to_serializable(self):
        return {'token_to_idx': self.token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def add_tokens(self, tokens):
        return [self.add_token[token] for token in tokens]

    def lookup_token(self, token):
        if token not in self.token_to_idx:
            raise KeyError("the token (%s) is not in the Vocabulary" % token)
        return self.token_to_idx[token]

    def lookup_index(self, index):
        if index not in self.idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self.idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self.token_to_idx)

    
### Test: Vocabulary instance ###
#label_vocab = Vocabulary()
#for index, row in df.iterrows():
#    label_vocab.add_token(row.label)
#print(label_vocab) # __str__
#print(len(label_vocab)) # __len__
#index = label_vocab.lookup_token(0)
#print(index) # 0 (label): 1 (index)
#print(label_vocab.lookup_index(index))


#%% ==================================== Sequence vocabulary ====================================
class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        super(SequenceVocabulary, self).__init__(token_to_idx)

        self.mask_token = mask_token
        self.unk_token = unk_token
        self.begin_seq_token = begin_seq_token
        self.end_seq_token = end_seq_token

        self.mask_index = self.add_token(self.mask_token)
        self.unk_index = self.add_token(self.unk_token)
        self.begin_seq_index = self.add_token(self.begin_seq_token)
        self.end_seq_index = self.add_token(self.end_seq_token)
        
        # Index to token
        self.idx_to_token = {idx: token \
                             for token, idx in self.token_to_idx.items()}

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self.unk_token,
                         'mask_token': self.mask_token,
                         'begin_seq_token': self.begin_seq_token,
                         'end_seq_token': self.end_seq_token})
        return contents

    def lookup_token(self, token):
        return self.token_to_idx.get(token, self.unk_index)
    
    def lookup_index(self, index):
        if index not in self.idx_to_token:
            raise KeyError("the index (%d) is not in the SequenceVocabulary" % index)
        return self.idx_to_token[index]
    
    def __str__(self):
        return "<SequenceVocabulary(size=%d)>" % len(self.token_to_idx)

    def __len__(self):
        return len(self.token_to_idx)


### Test: Get word counts ###+
#word_counts = Counter()
#for text in split_df.text:
#    for token in text.split(" "):
#        if token not in string.punctuation:
#            word_counts[token] += 1
            
### Create SequenceVocabulary instance ###
#doc_vocab = SequenceVocabulary()
#for word, word_count in word_counts.items():
#    if word_count >= args.cutoff:
#        doc_vocab.add_token(word)
#print(doc_vocab) # __str__
#print(len(doc_vocab)) # __len__
#index = doc_vocab.lookup_token("random")
#print(index)
#print(doc_vocab.lookup_index(index))


#%% ==================================== Vectorizer ====================================
class PapersVectorizer(object):
    def __init__(self, paper_vocab, label_vocab):
        self.paper_vocab = paper_vocab
        self.label_vocab = label_vocab

    def vectorize(self, paper):
        indices = [self.paper_vocab.lookup_token(token) for token in paper.split(" ")]
        indices = [self.paper_vocab.begin_seq_index] + indices + \
            [self.paper_vocab.end_seq_index]
        
        # Create vector
        paper_length = len(indices)
        vector = np.zeros(paper_length, dtype=np.int64)
        vector[:len(indices)] = indices
        return vector
    
    def unvectorize(self, vector):
        tokens = [self.paper_vocab.lookup_index(index) for index in vector]
        paper = " ".join(token for token in tokens)
        return paper

    @classmethod
    def from_dataframe(cls, df, cutoff):
        
        # Create class vocab
        label_vocab = Vocabulary()        
        for label in sorted(set(df.label)):
            label_vocab.add_token(label)

        # Get word counts
        word_counts = Counter()
        for paper in df.text:
            for token in paper.split(" "):
                word_counts[token] += 1
        
        # Create paper vocab
        paper_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                paper_vocab.add_token(word)
        
        return cls(paper_vocab, label_vocab)

    @classmethod
    def from_serializable(cls, contents):
        paper_vocab = SequenceVocabulary.from_serializable(contents['paper_vocab'])
        label_vocab = Vocabulary.from_serializable(contents['label_vocab'])
        return cls(paper_vocab=paper_vocab, label_vocab=label_vocab)
    
    def to_serializable(self):
        return {'paper_vocab': self.paper_vocab.to_serializable(),
                'label_vocab': self.label_vocab.to_serializable()}


# Vectorizer instance
#vectorizer = PapersVectorizer.from_dataframe(split_df, cutoff=args.cutoff)
#print(vectorizer.paper_vocab)
#print(vectorizer.label_vocab)
#vectorized_paper = vectorizer.vectorize(preprocess_text("Understanding species interactions can lead to a more holistic approach to the management of marine fisheries"))
#print(np.shape(vectorized_paper))
#print(vectorized_paper)
#print(vectorizer.unvectorize(vectorized_paper))



#%% ==================================== Dataset class ====================================
class PapersDataset(Dataset):
    def __init__(self, df, vectorizer):
        self.df = df
        self.vectorizer = vectorizer
        
        # Max paper length
        get_length = lambda paper: len(paper.split(" "))
        self.max_seq_length = max(map(get_length, df.text)) + 2 # (<BEGIN> + <END>)

        # Data splits
        self.train_df = self.df[self.df.split=='train']
        self.train_size = len(self.train_df)
        self.val_df = self.df[self.df.split=='val']
        self.val_size = len(self.val_df)
        self.test_df = self.df[self.df.split=='test']
        self.test_size = len(self.test_df)
        self.lookup_dict = {'train': (self.train_df, self.train_size), 
                            'val': (self.val_df, self.val_size),
                            'test': (self.test_df, self.test_size)}
        self.set_split('train')

        # Class weights (for imbalances)
        class_counts = df.label.value_counts().to_dict()
        def sort_key(item):
            return self.vectorizer.label_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, df, cutoff):
        train_df = df[df.split=='train']
        return cls(df, PapersVectorizer.from_dataframe(train_df, cutoff))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, df, vectorizer_filepath):
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(df, vectorizer)

    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return PapersVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self.vectorizer.to_serializable(), fp)

    def set_split(self, split="train"):
        self.target_split = split
        self.target_df, self.target_size = self.lookup_dict[split]

    def __str__(self):
        return "<Dataset(split={0}, size={1})".format(
            self.target_split, self.target_size)

    def __len__(self):
        return self.target_size

    def __getitem__(self, index):
        row = self.target_df.iloc[index]
        paper_vector = self.vectorizer.vectorize(row.text)
        label_index = self.vectorizer.label_vocab.lookup_token(row.label)
        return {'paper': paper_vector, 'label': label_index}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def generate_batches(self, batch_size, collate_fn, shuffle=True, drop_last=False, device="cpu"):
        dataloader = DataLoader(dataset=self, batch_size=batch_size,
                                collate_fn=collate_fn, shuffle=shuffle, 
                                drop_last=drop_last)
        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict



# Dataset instance
#dataset = PapersDataset.load_dataset_and_make_vectorizer(df=split_df, cutoff=args.cutoff)
#print(dataset) # __str__
#paper_vector = dataset[5]['paper'] # __getitem__
#print(paper_vector)
#print(dataset.vectorizer.unvectorize(paper_vector))
#print(dataset.class_weights)