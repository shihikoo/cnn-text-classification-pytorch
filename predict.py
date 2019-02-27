#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:30:41 2019
ref: github/GokuMohandas/practicalAI: notebooks/11_Convolutional_Neural_Networks.ipynb
@author: qwang
"""
import os
os.chdir('/home/qwang/rob')

import json
from argparse import Namespace

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models.keyedvectors import KeyedVectors

import data_process
from data_process import df, split_df, args, Vocabulary, SequenceVocabulary, PapersVectorizer, PapersDataset, preprocess_text
import model
from model import PapersModel


#%%
class Predict(object):
    def __init__(self, model, vectorizer, device="cpu"):
        self.model = model.to(device)
        self.vectorizer = vectorizer
        self.device = device
  
    def predict_rob(self, dataset):
        # Batch generator
        batch_generator = dataset.generate_batches(batch_size=len(dataset), shuffle=False, device=self.device)
        self.model.eval()
        
        # Predict
        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output
            y_pred =  self.model(batch_dict['paper'], apply_softmax=True)

            # Top k labels
            y_prob, indices = torch.topk(y_pred, k=len(self.vectorizer.label_vocab))
            probabilities = y_prob.detach().to('cpu').numpy()[0]
            indices = indices.detach().to('cpu').numpy()[0]

            results = []
            for probability, index in zip(probabilities, indices):
                rob_label = self.vectorizer.label_vocab.lookup_index(index)
                results.append({'RoB reporting': rob_label, 'probability': probability})

        return results
    

# Load vectorizer
with open(args.vectorizer_file) as fp:
    vectorizer = PapersVectorizer.from_serializable(json.load(fp))

# Use embeddings
args.use_med_embeddings = True
med_w2v = KeyedVectors.load_word2vec_format('wordvec/wikipedia-pubmed-and-PMC-w2v.bin', binary=True)
def make_embeddings_matrix(words):
    embedding_dim = args.embedding_dim
    embeddings = np.zeros((len(words), embedding_dim))
    for i, word in enumerate(words):
        if word in med_w2v:
            embedding_vector = med_w2v[word]
            embeddings[i, :] = embedding_vector[:embedding_dim]            
        else:
            embedding_i = torch.zeros(1, embedding_dim)
            nn.init.xavier_uniform_(embedding_i)
            embeddings[i, :] = embedding_i
    return embeddings

# Create embeddings
embeddings = None
if args.use_med_embeddings:
    words = vectorizer.paper_vocab.token_to_idx.keys()
    embeddings = make_embeddings_matrix(words=words)
    print ("<Embeddings(words={0}, dim={1})>".format(np.shape(embeddings)[0], np.shape(embeddings)[1])) 
del(med_w2v)

 
# Load the model
model = PapersModel(embedding_dim=args.embedding_dim, 
                    num_embeddings=len(vectorizer.paper_vocab),
                    num_input_channels=args.embedding_dim, 
                    num_channels=args.num_filters,
                    hidden_dim=args.hidden_dim,
                    num_classes=len(vectorizer.label_vocab),
                    dropout_p=args.dropout_p,
                    pretrained_embeddings=embeddings,
                    padding_idx=vectorizer.paper_vocab.mask_index)
model.load_state_dict(torch.load(args.model_state_file))
print(model.named_modules)


# Initialize
predict = Predict(model=model, vectorizer=vectorizer, device=args.device)


class PredictDataset(Dataset):
    def __init__(self, df, vectorizer):
        self.df = df
        self.vectorizer = vectorizer
        self.target_size = len(self.df)

    def __str__(self):
        return "<Dataset(size={1})>".format(self.target_size)

    def __len__(self):
        return self.target_size

    def __getitem__(self, index):
        row = self.df.iloc[index]
        paper_vector = self.vectorizer.vectorize(row.paper)
        return {'paper': paper_vector}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def generate_batches(self, batch_size, shuffle=True, drop_last=False, device="cpu"):
        dataloader = DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict


# Prediction
paper = input("Input the text to check its RoB reporting: ")
pred_df = pd.DataFrame([paper], columns=['paper'])

pred_df.paper = pred_df.paper.apply(preprocess_text)
pred_dataset = PredictDataset(pred_df, vectorizer)
results = predict.predict_rob(dataset=pred_dataset)
results
