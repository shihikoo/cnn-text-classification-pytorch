# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:32:05 2019
ref: github/GokuMohandas/practicalAI: notebooks/12_Embeddings.ipynb
@author: qwang
"""

import os
os.chdir('/home/qwang/rob')

import copy
import json
import time
from argparse import Namespace

import numpy as np
import matplotlib.pyplot as plt

from gensim.models.keyedvectors import KeyedVectors

import torch
import torch.nn as nn
import torch.optim as optim



import data_process
from data_process import df, split_df, args, Vocabulary, SequenceVocabulary, PapersVectorizer, PapersDataset
import model
from model import PapersModel


class Trainer(object):
    def __init__(self, dataset, model, model_state_file, save_dir, device, shuffle, 
               num_epochs, batch_size, learning_rate, early_stopping_criteria):
        self.dataset = dataset
        self.class_weights = dataset.class_weights.to(device)
        self.model = model.to(device)
        self.save_dir = save_dir
        self.device = device
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.loss_func = nn.CrossEntropyLoss(self.class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, mode='min', factor=0.5, patience=1)
        self.train_state = {
            'done_training': False,
            'stop_early': False, 
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'early_stopping_criteria': early_stopping_criteria,
            'learning_rate': learning_rate,
            'epoch_index': 0,
            
            'train_loss': [],
            'train_acc': [],
            'train_sens': [],
            'train_spec': [],
            'train_prec': [],
            'train_f1': [],
            
            'val_loss': [],
            'val_acc': [],
            'val_sens': [],
            'val_spec': [],
            'val_prec': [],
            'val_f1': [],
            
            'test_loss': -1,
            'test_acc': -1,
            'test_sens': -1,
            'test_spec': -1,
            'test_prec': -1,
            'test_f1': -1,
            
            'model_filename': model_state_file}
    
    def update_train_state(self):

        # Verbose
        print ("[Epoch{0} / Train] | LR {1} | loss: {2:.3f} | accuracy: {3:.2f}% | sensitivity: {4:.2f}% | specificity: {5:.2f}% | precision: {6:.2f}% | f1: {7:.2f}%".format(
          self.train_state['epoch_index'], self.train_state['learning_rate'], 
            self.train_state['train_loss'][-1], self.train_state['train_acc'][-1], 
            self.train_state['train_sens'][-1], self.train_state['train_spec'][-1], 
            self.train_state['train_prec'][-1], self.train_state['train_f1'][-1]))
        
        print ("[Epoch{0} / Val] | LR {1} | loss: {2:.3f} | accuracy: {3:.2f}% | sensitivity: {4:.2f}% | specificity: {5:.2f}% | precision: {6:.2f}% | f1: {7:.2f}%".format(
          self.train_state['epoch_index'], self.train_state['learning_rate'], 
            self.train_state['val_loss'][-1], self.train_state['val_acc'][-1], 
            self.train_state['val_sens'][-1], self.train_state['val_spec'][-1], 
            self.train_state['val_prec'][-1], self.train_state['val_f1'][-1]))
        

#        print ("[Epoch {0}] | LR: {1} | [Train LOSS]: {2:.2f} | [TRAIN ACC]: {3:.2f}% | [VAL LOSS]: {4:.2f} | [VAL ACC]: {5:.2f}%".format(
#          self.train_state['epoch_index'], self.train_state['learning_rate'], 
#            self.train_state['train_loss'][-1], self.train_state['train_acc'][-1], 
#            self.train_state['val_loss'][-1], self.train_state['val_acc'][-1]))
#
#        print ("[Epoch]: {0} | [LR]: {1} | [TRAIN SENS]: {2:.2f}% | [TRAIN SPEC]: {3:.2f}% | [TRAIN PREC]: {4:.2f}% | [TRAIN F1]: {5:.2f}%".format(
#          self.train_state['epoch_index'], self.train_state['learning_rate'], 
#            self.train_state['train_sens'][-1], self.train_state['train_spec'][-1], 
#            self.train_state['train_prec'][-1], self.train_state['train_f1'][-1]))
#        
#        print ("[Epoch]: {0} | [LR]: {1} | [VAL SENS]: {2:.2f}% | [VAL SPEC]: {3:.2f}% | [VAL PREC]: {4:.2f}% | [VAL F1]: {5:.2f}%".format(
#          self.train_state['epoch_index'], self.train_state['learning_rate'], 
#            self.train_state['val_sens'][-1], self.train_state['val_spec'][-1], 
#            self.train_state['val_prec'][-1], self.train_state['val_f1'][-1]))
        
        # Save one model at least
        if self.train_state['epoch_index'] == 0:
            torch.save(self.model.state_dict(), self.train_state['model_filename'])
            self.train_state['stop_early'] = False

        # Save model if performance improved
        elif self.train_state['epoch_index'] >= 1:
            loss_tm1, loss_t = self.train_state['val_loss'][-2:]

            # If loss worsened
            if loss_t >= self.train_state['early_stopping_best_val']:
                # Update step
                self.train_state['early_stopping_step'] += 1

            # Loss decreased
            else:
                # Save the best model
                if loss_t < self.train_state['early_stopping_best_val']:
                    torch.save(self.model.state_dict(), self.train_state['model_filename'])

                # Reset early stopping step
                self.train_state['early_stopping_step'] = 0

            # Stop early ?
            self.train_state['stop_early'] = self.train_state['early_stopping_step'] \
              >= self.train_state['early_stopping_criteria']
        return self.train_state
  
    def compute_accuracy(self, y_pred, y_target):
        _, y_pred_indices = y_pred.max(dim=1)        
        n_correct = torch.eq(y_pred_indices, y_target).sum().item()
        return n_correct / len(y_pred_indices) * 100
    
    def compute_sensitivity(self, y_pred, y_target):
        _, y_pred_indices = y_pred.max(dim=1)
        temp = torch.eq(y_pred_indices, torch.ones_like(y_target)) & torch.eq(y_target, torch.ones_like(y_target))
        tp = temp.sum().item()
        temp = torch.eq(y_pred_indices, torch.zeros_like(y_target)) & torch.eq(y_target, torch.ones_like(y_target))
        fn = temp.sum().item()
        if tp+fn == 0:
            sensitivity = 0
        else:
            sensitivity = 100 * tp / (tp+fn)
        return sensitivity  
    
    def compute_specificity(self, y_pred, y_target):
        _, y_pred_indices = y_pred.max(dim=1)
        temp = torch.eq(y_pred_indices, torch.zeros_like(y_target)) & torch.eq(y_target, torch.zeros_like(y_target))
        tn = temp.sum().item()
        temp = torch.eq(y_pred_indices, torch.ones_like(y_target)) & torch.eq(y_target, torch.zeros_like(y_target))
        fp = temp.sum().item()
        if tn+fp == 0:
            specificity = 0
        else:
            specificity = 100 * tn / (tn+fp)
        return specificity 
    
    def compute_precision(self, y_pred, y_target):
        _, y_pred_indices = y_pred.max(dim=1)
        temp = torch.eq(y_pred_indices, torch.ones_like(y_target)) & torch.eq(y_target, torch.ones_like(y_target))
        tp = temp.sum().item()
        temp = torch.eq(y_pred_indices, torch.ones_like(y_target)) & torch.eq(y_target, torch.zeros_like(y_target))
        fp = temp.sum().item()
        if tp+fp == 0:
            precision = 0
        else:
            precision = 100 * tp / (tp+fp) 
        return precision
    
    def compute_f1(self, y_pred, y_target):
        _, y_pred_indices = y_pred.max(dim=1)
        temp = torch.eq(y_pred_indices, torch.ones_like(y_target)) & torch.eq(y_target, torch.ones_like(y_target))
        tp = temp.sum().item()
        temp = torch.eq(y_pred_indices, torch.ones_like(y_target)) & torch.eq(y_target, torch.zeros_like(y_target))
        fp = temp.sum().item()
        temp = torch.eq(y_pred_indices, torch.zeros_like(y_target)) & torch.eq(y_target, torch.ones_like(y_target))
        fn = temp.sum().item()
        if 2*tp+fp+fn == 0:
            f1 = 0
        else:
            f1 = 100 * 2*tp / (2*tp+fp+fn)
        return f1 
    
    
    def pad_seq(self, seq, length):
        vector = np.zeros(length, dtype=np.int64)
        if len(seq) <= length:
            vector[:len(seq)] = seq
            vector[len(seq):] = self.dataset.vectorizer.paper_vocab.mask_index
        else:
            vector = seq[:length]
        
        return vector
    
    def collate_fn(self, batch):
        
        # Make a deep copy
        batch_copy = copy.deepcopy(batch)
        processed_batch = {"paper": [], "label": []}
        
        # Get max sequence length
        max_seq_len = args.max_seq_len ##########max([len(sample["paper"]) for sample in batch_copy])
        
        # Pad
        for i, sample in enumerate(batch_copy):
            seq = sample["paper"]
            label = sample["label"]
            padded_seq = self.pad_seq(seq, max_seq_len)
            processed_batch["paper"].append(padded_seq)
            processed_batch["label"].append(label)
            
        # Convert to appropriate tensor types
        processed_batch["paper"] = torch.LongTensor(processed_batch["paper"])
        processed_batch["label"] = torch.LongTensor(processed_batch["label"])
        
        return processed_batch    
  
    def run_train_loop(self):
        for epoch_index in range(self.num_epochs):
            self.train_state['epoch_index'] = epoch_index
      
            # Iterate over train dataset

            # initialize batch generator, set loss and acc to 0, set train mode on
            self.dataset.set_split('train')
            batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size, collate_fn=self.collate_fn, 
                shuffle=self.shuffle, device=self.device)
            running_loss = 0.0
            running_acc = 0.0
            running_sens = 0.0
            running_spec = 0.0
            running_prec = 0.0
            running_f1 = 0.0
            
            self.model.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                # zero the gradients
                self.optimizer.zero_grad()

                # compute the output
                y_pred = self.model(batch_dict['paper'])

                # compute the loss
                loss = self.loss_func(y_pred, batch_dict['label'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute gradients using loss
                loss.backward()

                # use optimizer to take a gradient step
                self.optimizer.step()
                
                # compute the accuracy
                acc_t = self.compute_accuracy(y_pred, batch_dict['label'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                
                # compute the sensitivity
                sens_t = self.compute_sensitivity(y_pred, batch_dict['label'])
                running_sens += (sens_t - running_sens) / (batch_index + 1)

                # compute the specificity
                spec_t = self.compute_specificity(y_pred, batch_dict['label'])
                running_spec += (spec_t - running_spec) / (batch_index + 1)
                
                # compute the precision
                prec_t = self.compute_precision(y_pred, batch_dict['label'])
                running_prec += (prec_t - running_prec) / (batch_index + 1)
                
                # compute the f1 score
                f1_t = self.compute_f1(y_pred, batch_dict['label'])
                running_f1 += (f1_t - running_f1) / (batch_index + 1)
                

            self.train_state['train_loss'].append(running_loss)
            self.train_state['train_acc'].append(running_acc)
            self.train_state['train_sens'].append(running_sens)
            self.train_state['train_spec'].append(running_spec)
            self.train_state['train_prec'].append(running_prec)
            self.train_state['train_f1'].append(running_f1)

            # Iterate over val dataset

            # initialize batch generator, set loss and acc to 0; set eval mode on
            self.dataset.set_split('val')
            batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size, collate_fn=self.collate_fn, 
                shuffle=self.shuffle, device=self.device)
            running_loss = 0.0
            running_acc = 0.0
            running_sens = 0.0
            running_spec = 0.0
            running_prec = 0.0
            running_f1 = 0.0
            
            self.model.eval()

            for batch_index, batch_dict in enumerate(batch_generator):

                # compute the output
                y_pred =  self.model(batch_dict['paper'])

                # compute the loss
                loss = self.loss_func(y_pred, batch_dict['label'])
                loss_t = loss.to("cpu").item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                acc_t = self.compute_accuracy(y_pred, batch_dict['label'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                
                # compute the sensitivity
                sens_t = self.compute_sensitivity(y_pred, batch_dict['label'])
                running_sens += (sens_t - running_sens) / (batch_index + 1)
                
                # compute the specificity
                spec_t = self.compute_specificity(y_pred, batch_dict['label'])
                running_spec += (spec_t - running_spec) / (batch_index + 1)
                
                # compute the precision
                prec_t = self.compute_precision(y_pred, batch_dict['label'])
                running_prec += (prec_t - running_prec) / (batch_index + 1)
                
                # compute the f1 score
                f1_t = self.compute_f1(y_pred, batch_dict['label'])
                running_f1 += (f1_t - running_f1) / (batch_index + 1)


            self.train_state['val_loss'].append(running_loss)
            self.train_state['val_acc'].append(running_acc)
            self.train_state['val_sens'].append(running_sens)
            self.train_state['val_spec'].append(running_spec)
            self.train_state['val_prec'].append(running_prec)
            self.train_state['val_f1'].append(running_f1)
            
            
            self.train_state = self.update_train_state()
            self.scheduler.step(self.train_state['val_loss'][-1])
            if self.train_state['stop_early']:
                break
          
    def run_test_loop(self):
        # initialize batch generator, set loss and acc to 0; set eval mode on
        self.dataset.set_split('test')
        batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size, collate_fn=self.collate_fn, 
                shuffle=self.shuffle, device=self.device)
        running_loss = 0.0
        running_acc = 0.0
        running_sens = 0.0
        running_spec = 0.0
        running_prec = 0.0
        running_f1 = 0.0
            
        self.model.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output
            y_pred =  self.model(batch_dict['paper'])

            # compute the loss
            loss = self.loss_func(y_pred, batch_dict['label'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = self.compute_accuracy(y_pred, batch_dict['label'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # compute the sensitivity
            sens_t = self.compute_sensitivity(y_pred, batch_dict['label'])
            running_sens += (sens_t - running_sens) / (batch_index + 1)
            
            # compute the specificity
            spec_t = self.compute_specificity(y_pred, batch_dict['label'])
            running_spec += (spec_t - running_spec) / (batch_index + 1)
                
            # compute the precision
            prec_t = self.compute_precision(y_pred, batch_dict['label'])
            running_prec += (prec_t - running_prec) / (batch_index + 1)
                
            # compute the f1 score
            f1_t = self.compute_f1(y_pred, batch_dict['label'])
            running_f1 += (f1_t - running_f1) / (batch_index + 1)


        self.train_state['test_loss'] = running_loss
        self.train_state['test_acc'] = running_acc
        self.train_state['test_sens'] = running_sens
        self.train_state['test_spec'] = running_spec
        self.train_state['test_prec'] = running_prec
        self.train_state['test_f1'] = running_f1

    
    def plot_performance(self):
        # Figure size
        plt.figure(figsize=(15,5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(trainer.train_state["train_loss"], label="train")
        plt.plot(trainer.train_state["val_loss"], label="val")
        plt.legend(loc='upper right')

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.plot(trainer.train_state["train_acc"], label="train")
        plt.plot(trainer.train_state["val_acc"], label="val")
        plt.legend(loc='lower right')

        # Save figure
        plt.savefig(os.path.join(self.save_dir, "performance.png"))

        # Show plots
        plt.show()
    
    def save_train_state(self):
        self.train_state["done_training"] = True
        with open(os.path.join(self.save_dir, "train_state.json"), "w") as fp:
            json.dump(self.train_state, fp)


#%% Using embeddings
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



args.use_med_embeddings = True

#%% Initialization
dataset = PapersDataset.load_dataset_and_make_vectorizer(df=split_df, cutoff=args.cutoff)
dataset.save_vectorizer(args.vectorizer_file)
vectorizer = dataset.vectorizer

# Create embeddings
embeddings = None
if args.use_med_embeddings:
    words = vectorizer.paper_vocab.token_to_idx.keys()
    embeddings = make_embeddings_matrix(words=words)
    print ("<Embeddings(words={0}, dim={1})>".format(np.shape(embeddings)[0], np.shape(embeddings)[1])) 

del(med_w2v)

# Initialize model 
model = PapersModel(embedding_dim=args.embedding_dim, 
                    num_embeddings=len(vectorizer.paper_vocab), 
                    num_input_channels=args.embedding_dim, 
                    num_channels=args.num_filters, hidden_dim=args.hidden_dim, 
                    num_classes=len(vectorizer.label_vocab), 
                    dropout_p=args.dropout_p, pretrained_embeddings=embeddings, # pretrained_embeddings=None, 
                    padding_idx=vectorizer.paper_vocab.mask_index)
print(model.named_modules)


#%% Train
trainer = Trainer(dataset=dataset, model=model, 
                  model_state_file=args.model_state_file, 
                  save_dir=args.save_dir, device=args.device,
                  shuffle=args.shuffle, num_epochs=args.num_epochs, 
                  batch_size=args.batch_size, learning_rate=args.learning_rate, 
                  early_stopping_criteria=args.early_stopping_criteria)


start_time = time.time()
trainer.run_train_loop()
elapsed_time = time.time() - start_time
print('Time elapsed: {0:.1f} minutes'.format(elapsed_time/60))

#%% Plot performance
trainer.plot_performance()


#%% Test performance
trainer.run_test_loop()

print ("[Test] loss: {0:.3f} | accuracy: {1:.2f}% | sensitivity: {2:.2f}% | specificity: {3:.2f}% | precision: {4:.2f}% | f1: {5:.2f}%".format(
         trainer.train_state['test_loss'], trainer.train_state['test_acc'], 
         trainer.train_state['test_sens'], trainer.train_state['test_spec'], 
         trainer.train_state['test_prec'], trainer.train_state['test_f1']))


#%% Save all results
trainer.save_train_state()


