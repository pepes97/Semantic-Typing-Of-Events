import re
import torch
import logging
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from utils import model_directory

class TrainerBART:
    def __init__(self, tokenizer, model, loss, device, model_path, type_model, max_len):
            self.device = device
            self.tokenizer = tokenizer
            self.model = model
            self.type_model = type_model
            self.loss = loss
            self.model_path = model_path
            self.max_len = max_len
            self.best_mrr = 0
            self.patience = 0
  
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

            
            validation_metrics, self.best_mrr, self.patience = self.evaluate(dev_dataloader, self.best_mrr, self.patience)
            
            keys = set()
            keys |= set(training_metrics.keys())
            keys |= set(validation_metrics.keys())

            table = []

            for key in keys:
                table.append((key, training_metrics.get(key, float('Nan')), validation_metrics.get(key, float('Nan'))))

            print(tabulate(table, headers=('metric', 'train', 'dev')) + '\n')

            if self.patience ==3:
                print("\033[1m No improvement for 3 epochs in a row, stop \033[0m \n")
                break
                
            logging.info('Complete')

    def evaluate(self,dev_dataloader, best_mrr, patience):
        self.model.eval()
        with torch.no_grad():
            recall1_verb =[]
            recall10_verb= []
            mrr_verb = []

            recall1_arg =[]
            recall10_arg = []
            mrr_arg = []

            iterator = tqdm(dev_dataloader)
            for batch in iterator:
                source = batch["source"].to(self.device)
                target = batch["target"].to(self.device)
                
                generate_batch = self.model.model.generate(source,max_length=175, num_beams=10,num_return_sequences=10,early_stopping=True)
                
                for i in range(len(target)):     
                    gold_elem = self.tokenizer.decode(target[i], skip_special_tokens=True)
                    predictions = generate_batch[i*10:i*10+10]
                    
                    new_predictions = []
                    for j in range(len(predictions)):
                        new_predictions.append(self.tokenizer.decode(predictions[j], skip_special_tokens=True))
                    

                    def test_verb(gold_elem, predictions):
                        found = False
                        pattern = r"{(.*?)}"
                        gold_verb = re.findall(pattern, gold_elem, flags=0)[0].strip()
                        
                        for idx,pred in enumerate(predictions):
                            pattern = r"{(.*?)}"
                            try:
                                pred_verb = re.findall(pattern, pred, flags=0)[0].strip()
                            except:
                                if idx==0:
                                    recall1_verb.append(0.)
                                continue
                            if idx == 0:
                                if pred_verb == gold_verb:
                                    recall1_verb.append(1.)
                                    recall10_verb.append(1.)
                                    mrr_verb.append(1.)
                                    found=True
                                    break
                                else:
                                    recall1_verb.append(0.)
                            else:
                                if pred_verb == gold_verb:
                                    recall10_verb.append(1.)
                                    mrr_verb.append(1./float(idx+1))
                                    found=True
                                    break

                        if found ==False:
                            recall10_verb.append(0.)
                            mrr_verb.append(0.)

                        return mrr_verb,recall1_verb,recall10_verb


                    def test_arg(gold_elem, predictions):
                        found = False
                        pattern = r"{{(.*?)}}"
                        try:
                            gold_arg = re.findall(pattern, gold_elem, flags=0)[0].strip()
                        except:
                            return mrr_arg,recall1_arg,recall10_arg
                    
                        for idx,pred in enumerate(predictions):
                            pattern = r"{{(.*?)}}"
                            try:
                                pred_arg = re.findall(pattern, pred, flags=0)[0].strip()
                            except:
                                if idx==0:
                                    recall1_arg.append(0.)
                                continue

                            if idx == 0:
                                if pred_arg == gold_arg:
                                    recall1_arg.append(1.)
                                    recall10_arg.append(1.)
                                    mrr_arg.append(1.)
                                    found=True
                                    break
                                else:
                                    recall1_arg.append(0.)
                            else:
                                if pred_arg == gold_arg:
                                    recall10_arg.append(1.)
                                    mrr_arg.append(1./float(idx+1))
                                    found=True
                                    break

                        if found ==False:
                            recall10_arg.append(0.)
                            mrr_arg.append(0.)

                        return mrr_arg,recall1_arg,recall10_arg

                    mrr_v, rec1v, rec10v = test_verb(gold_elem, new_predictions)
                    mrr_a,rec1a, rec10a = test_arg(gold_elem, new_predictions)
            
            metrics = {}

            metrics['mrr_verbs'] = np.average(mrr_v)
            metrics['recall@1_verbs'] = np.average(rec1v)
            metrics['recall@10_verbs'] = np.average(rec10v)

            metrics['mrr_args'] = np.average(mrr_a)
            metrics['recall@1_args'] = np.average(rec1a)
            metrics['recall@10_args'] = np.average(rec10a)

            if best_mrr < metrics['mrr_verbs']:
                best_mrr = metrics['mrr_verbs']
                patience = 0
                torch.save(self.model.state_dict(),self.model_path + "/bart_model_"+self.type_model+model_directory(self.model_path)+"_len"+str(self.max_len)+"_downsizing"+str(self.downsizing)+"k_action"+str(self.k_action)+"k_object"+str(self.k_object)+"_SEED10_lr2e-5.pt")
                with open(self.model_path + "/best_mrr_"+self.type_model+".txt", "w") as f:
                    f.write(str(best_mrr))
                with open(self.model_path + "/patience_"+self.type_model+".txt", "w") as f:
                    f.write(str(patience))
                print(f"\033[1m Performance improvement, model saved in {self.model_path} \033[0m \n")
            else:
                patience +=1
                with open(self.model_path + "/patience_"+self.type_model+".txt", "w") as f:
                    f.write(str(patience))

        return metrics, best_mrr, patience
        

    def prediction_final(self,test_dataloader):
        self.model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            recall1_verb =[]
            recall10_verb= []
            mrr_verb = []

            recall1_arg =[]
            recall10_arg = []
            mrr_arg = []

            n = 45 if self.max_len == 20 else 200

            iterator = tqdm(test_dataloader)
            for batch in iterator:
                source = batch["source"].to(self.device)
                target = batch["target"].to(self.device)

                try:
                    generate_batch = self.model.model.generate(source, max_length=n, num_beams=100,num_return_sequences=100, early_stopping=True)
                except:
                    continue
                for i in range(len(target)):  
                    verbs_pred = []
                    args_pred = []
                    
                    gold_elem = self.tokenizer.decode(target[i], skip_special_tokens=True)
                    predictions = generate_batch[i*100:i*100+100]
                    
                    new_predictions = []
                    for j in range(len(predictions)):
                        new_predictions.append(self.tokenizer.decode(predictions[j], skip_special_tokens=True))
                    
                    def find_verbs_args(gold_elem,new_predictions, verbs_pred, args_pred):

                        for idx,pred in enumerate(new_predictions):
                            pattern = r"{(.*?)}"
                            try: 
                                pred_verb = re.findall(pattern, pred, flags=0)[0].strip()
                                if pred_verb not in verbs_pred:
                                    verbs_pred.append(pred_verb)
                            except:
                                continue
                    

                        for idx,pred in enumerate(new_predictions):
                            pattern = r"{{(.*?)}}"
                            try:
                                pred_arg = re.findall(pattern, pred, flags=0)[0].strip()
                                if pred_arg not in args_pred:
                                    args_pred.append(pred_arg)
                            except:
                                continue

                        return verbs_pred, args_pred
                

                    def test_verb(gold_elem, verbs_pred):
                        
                        found = False
                        pattern = r"{(.*?)}"
                        gold_verb = re.findall(pattern, gold_elem, flags=0)[0].strip()
                        
                        for idx,pred_verb in enumerate(verbs_pred):
                            if idx == 0:
                                if pred_verb == gold_verb:
                                    recall1_verb.append(1.)
                                    recall10_verb.append(1.)
                                    mrr_verb.append(1.)
                                    found=True
                                    break
                                else:
                                    recall1_verb.append(0.)
                            else:
                                if idx < 10:
                                    if pred_verb == gold_verb:
                                        recall10_verb.append(1.)
                                        mrr_verb.append(1./float(idx+1))
                                        found=True
                                        break
                                else:
                                    if pred_verb == gold_verb:
                                        mrr_verb.append(1./float(idx+1))
                                        recall10_verb.append(0.)
                                        found=True
                                        break


                        if found ==False:
                            recall10_verb.append(0.)
                            mrr_verb.append(0.)

                        return mrr_verb,recall1_verb,recall10_verb


                    def test_arg(gold_elem, args_pred):
                        found = False
                        pattern = r"{{(.*?)}}"
                        try:
                            gold_arg = re.findall(pattern, gold_elem, flags=0)[0].strip()
                        except:
                            return mrr_arg,recall1_arg,recall10_arg

                        
                        for idx,pred_arg in enumerate(args_pred):
                            if idx == 0:
                                if pred_arg == gold_arg:
                                    recall1_arg.append(1.)
                                    recall10_arg.append(1.)
                                    mrr_arg.append(1.)
                                    found=True
                                    break
                                else:
                                    recall1_arg.append(0.)
                            else:
                                if idx < 10:
                                    if pred_arg == gold_arg:
                                        recall10_arg.append(1.)
                                        mrr_arg.append(1./float(idx+1))
                                        found=True
                                        break
                                else:
                                    if pred_arg == gold_arg:
                                        mrr_arg.append(1./float(idx+1))
                                        recall10_arg.append(0.)
                                        found=True
                                        break

                        if found ==False:
                            recall10_arg.append(0.)
                            mrr_arg.append(0.)

                        return mrr_arg,recall1_arg,recall10_arg

                    start = 200
                    num_ret=200
                    verbs_pred, args_pred = find_verbs_args(gold_elem, new_predictions, verbs_pred, args_pred)
                    
                    while len(args_pred)<10:
                        new_predictions = []
                        try:
                            pred_new = self.model.model.generate(source[i].unsqueeze(0), max_length=n, num_beams=start,num_return_sequences=num_ret, early_stopping=True)
                        except:
                            break
                        for j in range(len(pred_new)):
                            new_predictions.append(self.tokenizer.decode(pred_new[j], skip_special_tokens=True))
                        start += 100
                        num_ret +=100
                        verbs_pred, args_pred = find_verbs_args(gold_elem, new_predictions, verbs_pred, args_pred)
                
                    mrr_v, rec1v, rec10v = test_verb(gold_elem, verbs_pred) 
                    mrr_a,rec1a, rec10a = test_arg(gold_elem, args_pred)

        return  mrr_v, rec1v, rec10v, mrr_a,rec1a, rec10a