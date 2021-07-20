from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class Seq2SeqDatasetBART(Dataset):
    def __init__(self, tokenizer, path_file, max_len):
        self.tokenizer = tokenizer
        self.path_file = path_file
        self.data = []
        self.max_len = max_len
        self.setup()

    def setup(self):

        with open(self.path_file) as f:
            for line in tqdm(f, desc='Reading lines'):
                line = line.strip()
                line = line.split("\t")
                title = line[1]
                split_title = title.split()

                for i in range(len(split_title)):
                    if "**" in split_title[i]:
                        split_title[i] = split_title[i].replace("**", "", 2)
                        split_title[i] = " { "+split_title[i]+" } "

                for i in range(len(split_title)):
                    if "*" in split_title[i]:
                        num_ast = split_title[i].count("*")
                        if num_ast==2:
                            split_title[i] = split_title[i].replace("*", "", 2)
                            split_title[i] = " {{ "+split_title[i]+" }} "
                        elif num_ast == 1:
                            index_ast = split_title[i].index("*")
                            if index_ast==0:
                                split_title[i] = split_title[i].replace("*", "",1)
                                split_title[i] = " {{ "+split_title[i]
                            else:
                                split_title[i] = split_title[i].replace("*", "",1)
                                split_title[i] = split_title[i]+" }} "

                new_title = " ".join(e for e in split_title)

                verb = line[2]
                verb_sense = line[3]

                arg = line[4]
                arg_sense = line[5]

                source = line[6:]
                marker = " : "
                point = "  . "

                encoded_source = []
                for i,elem in enumerate(source):
                    if i == len(source)-1:
                        encoded_source += [str(i+1)+"."] + [elem] + [point]
                    else:
                        encoded_source += [str(i+1)+"."] + [elem] + [point]

                if len(encoded_source)>1:
                    encoded_source = [" ".join(e for e in encoded_source)]

                if self.max_len==175:
                    target = [new_title + point + verb + marker + verb_sense + point + arg + marker + arg_sense + point]
                elif self.max_len == 20:
                    target = [new_title]
                    
                encoded_s,encoded_mask, encoded_t = self.encoded_sentences(encoded_source, target)

                data = {
                    "source": encoded_s,
                    "attention_mask": encoded_mask,
                    "target":encoded_t,
                }
                self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def encoded_sentences(self, source, target):

        encoded_source = self.tokenizer.encode(self.tokenizer.tokenize(source[0]), truncation = True)
        encoded_s = torch.tensor(encoded_source)

        encoded_target = self.tokenizer.encode(self.tokenizer.tokenize(target[0]))

        encoded_t = torch.tensor(encoded_target)
        encoded_mask = torch.ones((encoded_s.shape))

        return encoded_s,encoded_mask,encoded_t