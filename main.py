import logging
import numpy as np
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os

import sys
import re
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
from torch.optim  import Adam
import argparse

from dataset import Seq2SeqDatasetBART
from model import Seq2SeqModelBART, HyperparamsBART
from trainer import TrainerBART
from utils import model_directory, find_files

np.random.seed(10)

def pad(samples):
    
    batch_size = len(samples)
    max_length = max([len(sample) for sample in samples])

    batch = torch.ones((batch_size, max_length), dtype=torch.int64)

    for i in range(len(samples)):
        for j in range(len(samples[i])):
            batch[i, j] = samples[i][j]

    return batch

def pad_mask(samples):
    
    batch_size = len(samples)
    max_length = max([len(sample) for sample in samples])

    batch = torch.zeros((batch_size, max_length), dtype=torch.int64)

    for i in range(len(samples)):
        for j in range(len(samples[i])):
            batch[i, j] = samples[i][j]

    return batch

def collate_fn(samples):
    keys = samples[0].keys()
    dictionary = {}
    for key in keys:
      lists= []
      if key != "attention_mask":
        for sample in samples:
          lists.append(sample[key])
        padding = pad(lists)
        dictionary[key] = padding
      else:
        for sample in samples:
          lists.append(sample[key])
        padding = pad_mask(lists)
        dictionary[key] = padding
    return dictionary


def main(model_path, only_test, type_model, batch_size, max_len):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set, dev_set, test_set = find_files(model_path)
    print(f"\033[1mTrain file \033[0m: {train_set} \033[0m")

    print(f"\033[1mModel BART: {type_model} \033[0m")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-'+type_model, add_prefix_space=True, force_bos_token_to_be_generated=True)
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-"+type_model)
    
    print("\033[1mCreating train dataset \033[0m")
    train_dataset = Seq2SeqDatasetBART(tokenizer=tokenizer, path_file=train_set, max_len=max_len)
    print("\033[1mCreating dev dataset \033[0m")
    dev_dataset = Seq2SeqDatasetBART(tokenizer=tokenizer, path_file=dev_set, max_len=max_len)
    print("\033[1mCreating test dataset \033[0m")
    test_dataset = Seq2SeqDatasetBART(tokenizer=tokenizer, path_file=test_set, max_len=max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print("\033[1mCreating model \033[0m \n")
    params = HyperparamsBART()
    seq2seq = Seq2SeqModelBART(tokenizer=tokenizer, model=bart_model, hparams=params).to(device)

    trainer = TrainerBART(tokenizer,seq2seq,torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id), device, model_path, type_model, max_len)
    if  os.path.isfile(model_path + "/bart_model_"+type_model+model_directory(model_path)+"_len"+max_len+"_SEED10_lr2e-5.pt"):
        trainer.model.load_state_dict(torch.load(model_path + "/bart_model_"+type_model+model_directory(model_path)+"_len"+max_len+"_SEED10_lr2e-5.pt"))
        with open(model_path + "/best_mrr_"+type_model+".txt") as f:
          trainer.best_mrr = float(f.readline().rstrip())
        with open(model_path + "/patience_"+type_model+".txt") as f:
          trainer.patience = int(f.readline().rstrip())
        print("\033[1mModel loaded \033[0m \n")
    
    if not only_test:
        print("\033[1m\033[94m Start training... \033[0m \n")
        trainer.training(optim.Adam(seq2seq.parameters(), lr=2e-5), train_dataloader, dev_dataloader, 10)
        print("\033[1m\033[92m Testing... \033[0m \n")
        trainer.model.load_state_dict(torch.load(model_path + "/bart_model_"+type_model+model_directory(model_path)+"_len"+max_len+"_SEED10_lr2e-5.pt"))
        mrr_v, rec1v, rec10v, mrr_a,rec1a, rec10a = trainer.prediction_test(test_dataloader)

    else:
        print("\033[1m\033[92m Testing... \033[0m \n")
        mrr_v, rec1v, rec10v, mrr_a,rec1a, rec10a = trainer.prediction_test(test_dataloader)
   
    print(f"\033[1m***** VERBS ***** -> MRR: {str(np.average(mrr_v))}, RECALL@1: {str(np.average(rec1v))}, RECALL@10: {str(np.average(rec10v))} \033[0m \n")
    print(f"\033[1m***** ARGS ****** -> MRR: {str(np.average(mrr_a))}, RECALL@1: {str(np.average(rec1a))}, RECALL@10: {str(np.average(rec10a))} \033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="the path where you want to save the model, you can to choose between: 'models/their_split', 'models/their_split_mixed', 'models/their_split_all_processes', 'models/their_split_WORD'")
    parser.add_argument("--type-model", type=str, required=True, 
                        help="the model of BART, you can to choose between 'base' or 'large'")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="size of the batch")
    parser.add_argument("--only-test", type=str, default=False,
                        help="flag for only testing the model")
    parser.add_argument("--max-len", type=int, default= 175,
                        help="max length of sentences in generation phase")  
    
    args = parser.parse_args()
    model_path = args.model_path
    only_test = args.only_test
    type_model = args.type_model
    batch_size = args.batch_size
    max_len = args.max_len

    main(model_path, only_test, type_model, batch_size, max_len)