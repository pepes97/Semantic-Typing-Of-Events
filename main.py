import os
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration

from trainer import TrainerBART
from dataset import Seq2SeqDatasetBART
from model import Seq2SeqModelBART, HyperparamsBART
from utils import model_directory, find_files, collate_fn

np.random.seed(10)


def main(model_path, only_test, type_model, batch_size, max_len):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\033[1mDevice \033[0m: {device} \033[0m")
    train_set, dev_set, test_set = find_files(model_path)

    print(f"\033[1mTrain file \033[0m: {train_set} \033[0m")
    print(f"\033[1mDev file \033[0m: {dev_set} \033[0m")
    print(f"\033[1mTest file \033[0m: {test_set} \033[0m")

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
    print(f"Len Train: {len(train_dataloader)*batch_size}")
    
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Len Dev: {len(dev_dataloader)*batch_size}")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Len Test: {len(test_dataloader)*batch_size}")
    print("\033[1mCheck Dataloader \033[0m")
    for batch in train_dataloader:
      for source, target in zip(batch['source'][: 25], batch['target'][: 25]):
        print("**SOURCE**")
        print(tokenizer.decode(source, skip_special_tokens=True).split())
        print("***GOLD***")
        print(tokenizer.decode(target,skip_special_tokens=True).split())
      break
    print()

    print("\033[1mCreating model \033[0m \n")
    params = HyperparamsBART()
    seq2seq = Seq2SeqModelBART(tokenizer=tokenizer, model=bart_model, hparams=params).to(device)

    trainer = TrainerBART(tokenizer,seq2seq,torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id), device, model_path, type_model, max_len)
    
    if  os.path.isfile(model_path + "/bart_model_"+type_model+model_directory(model_path)+"_len"+str(max_len)+"_SEED10_lr2e-5.pt"):
        trainer.model.load_state_dict(torch.load(model_path + "/bart_model_"+type_model+model_directory(model_path)+"_len"+str(max_len)+"_SEED10_lr2e-5.pt", map_location=device))

        print("\033[1mModel loaded \033[0m \n")
  
    if not only_test:
        if os.path.exists(model_path + "/best_mrr_"+type_model+".txt"):
          with open(model_path + "/best_mrr_"+type_model+".txt") as f:
            trainer.best_mrr = float(f.readline().rstrip())
    
        if os.path.exists(model_path + "/patience_"+type_model+".txt"):
          with open(model_path + "/patience_"+type_model+".txt") as f:
            trainer.patience = int(f.readline().rstrip())

        print("\033[1m\033[94mStart training... \033[0m \n")
        trainer.training(optim.Adagrad(seq2seq.parameters(), lr=2e-5), train_dataloader, dev_dataloader, 15)
        print("\033[1m\033[92mTesting... \033[0m \n")
        trainer.model.load_state_dict(torch.load(model_path + "/bart_model_"+type_model+model_directory(model_path)+"_len"+str(max_len)+"_SEED10_lr2e-5.pt", map_location=device))
        
        mrr_v, rec1v, rec10v, mrr_a,rec1a, rec10a = trainer.prediction_final(test_dataloader)
    else:
        print("\033[1m\033[92m Testing... \033[0m \n")
        mrr_v, rec1v, rec10v, mrr_a,rec1a, rec10a = trainer.prediction_final(test_dataloader)
          
    print(f"\033[1m***** VERBS ***** -> MRR: {str(np.average(mrr_v))}, RECALL@1: {str(np.average(rec1v))}, RECALL@10: {str(np.average(rec10v))} \033[0m \n")
    print(f"\033[1m***** ARGS ****** -> MRR: {str(np.average(mrr_a))}, RECALL@1: {str(np.average(rec1a))}, RECALL@10: {str(np.average(rec10a))} \033[0m")
    
if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="the path where you want to save the model, you can to choose between: 'models/their_split', 'models/their_split_mixed', 'models/their_split_all_processes'")
    parser.add_argument("--type-model", type=str, required=True, 
                        help="the model of BART, you can to choose between 'base' or 'large'")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="size of the batch, default is 2")
    parser.add_argument("--only-test", type=str, default=False,
                        help="flag for only testing the model, default is False")
    parser.add_argument("--max-len", type=int, default= 20,
                        help="max length used to distinguish between with and without glosses, default is 20 refers to without glosses and 175 is with glosses")  
    

    args = parser.parse_args()
    model_path = args.model_path
    only_test = args.only_test
    type_model = args.type_model
    batch_size = args.batch_size
    max_len = args.max_len
    

    main(model_path, only_test, type_model, batch_size, max_len)