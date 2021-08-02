import logging
from utils import model_directory
from tabulate import tabulate
from tqdm import tqdm
import re
import torch
import numpy as np
from utils import model_directory
from sacrebleu import corpus_bleu

class TrainerBARTInverse():
  def __init__(self, tokenizer, model, loss, device, model_path, type_model, max_len):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        #self.model.load_state_dict(torch.load("/content/drive/MyDrive/SemanticTypingOfEvents/baseline/Seq2SeqBART/bart_model_base_all_processes_reverse_their_split_len200_SEED10_lr2e-5.pt"))
        self.loss = loss
        self.best_bleu = 0
        self.patience = 0
        self.type_model = type_model
        self.max_len = max_len
        self.model_path = model_path

  
  def training(self, optimizer,train_dataloader, dev_dataloader, epochs):

    logging.info('Starting training')
    steps_done = 0

    for epoch in range(epochs):
      logging.info(f'Epoch {epoch + 1}/{epochs}')
      self.model.train()
      train_loss_n = 0.0
      train_loss_d = 0
      iterator = tqdm(train_dataloader)
      for batch in iterator:
        source = batch["source"].to(self.device)
        target = batch["target"].to(self.device)

        decoder_input_ids = self.model.shift_tokens_right(target, self.tokenizer.pad_token_id)

        optimizer.zero_grad()
        logits=self.model(source, decoder_input_ids = decoder_input_ids)[0]
        
        logits = logits.reshape((logits.shape[0]*logits.shape[1],logits.shape[-1]))
        gold = target.view(-1)

        
        loss = self.loss(logits, gold)
        loss.backward()
        optimizer.step()

        train_loss_n += loss.item()
        train_loss_d += 1
        steps_done += 1

        iterator.set_postfix(
          loss=(train_loss_n / train_loss_d),
          epoch=epoch,
          step=steps_done
        )
        training_metrics = {}
        training_metrics['loss'] = train_loss_n / train_loss_d if train_loss_d > 0 else 0.0

      
      validation_metrics, self.best_bleu, self.patience = self.evaluate(dev_dataloader, self.best_bleu, self.patience)
      
      keys = set()
      keys |= set(training_metrics.keys())
      keys |= set(validation_metrics.keys())

      table = []

      for key in keys:
          table.append((key, training_metrics.get(key, float('Nan')), validation_metrics.get(key, float('Nan'))))

      print(tabulate(table, headers=('metric', 'train', 'dev')) + '\n')

      if self.patience ==5:
        print("\033[1m No improvement for 5 epochs in a row, stop \033[0m \n")
        break
        
    logging.info('Complete')

  def evaluate(self,dev_dataloader, best_bleu, patience):
    self.model.eval()
    
    bleu_score = []
    iterator = tqdm(dev_dataloader)
    for batch in iterator:
        source = batch["source"].to(self.device)
        target = batch["target"].to(self.device)
        
        n = 100 if self.max_len == 100 else 200
        try:
            generate_batch = self.model.model.generate(source,max_length=n, num_beams=10,num_return_sequences=1,early_stopping=True)
        except:
            continue
        decode_source = [self.tokenizer.decode(source[i], skip_special_tokens=True) for i in range(len(source))]
        gold_elem =  [self.tokenizer.decode(target[i], skip_special_tokens=True) for i in range(len(target))]
        predictions = [self.tokenizer.decode(generate_batch[i*1:i*1+1][0], skip_special_tokens=True) for i in range(len(target))]
        bleu_score.append(corpus_bleu(predictions, [gold_elem]).score)


    
    metrics = {}

    metrics['bleu_score'] = np.average(bleu_score)

    if best_bleu < metrics['bleu_score']:
        best_bleu = metrics['bleu_score']
        torch.save(self.model.state_dict(),self.model_path + "/bart_model_"+self.type_model+model_directory(self.model_path)+"_len"+str(self.max_len)+"_SEED10_lr2e-5.pt")
        with open(self.model_path + "/best_bleu_"+self.type_model+".txt", "w") as f:
            f.write(str(best_bleu))
        with open(self.model_path + "/patience_"+self.type_model+".txt", "w") as f:
            f.write(str(patience))
        print(f"\033[1m Performance improvement, model saved in {self.model_path} \033[0m \n")
    else:
        patience +=1
        with open(self.model_path + "/patience_"+self.type_model+".txt", "w") as f:
            f.write(str(patience))
    return metrics, best_bleu, patience

  def prediction_final(self,dev_dataloader):
        self.model.eval()
        
        bleu_score = []
        iterator = tqdm(dev_dataloader)
        for batch in iterator:
            source = batch["source"].to(self.device)
            target = batch["target"].to(self.device)
            
            generate_batch = self.model.model.generate(source,max_length=100, num_beams=10,num_return_sequences=1,early_stopping=True)
            decode_source = [self.tokenizer.decode(source[i], skip_special_tokens=True) for i in range(len(source))]
            gold_elem =  [self.tokenizer.decode(target[i], skip_special_tokens=True) for i in range(len(target))]
            predictions = [self.tokenizer.decode(generate_batch[i*1:i*1+1][0], skip_special_tokens=True) for i in range(len(target))]
            bleu_score.append(corpus_bleu(predictions, [gold_elem]).score)

            for i in range(len(target)):
              print("**GOLD PROCESSES**")
              print(gold_elem[i])
              print("**PREDICTION PROCESSES**")
              print(predictions[i])
        
             
        
        return bleu_score