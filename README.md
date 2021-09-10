# Semantic-Typing-Of-Events


# Setup

```
conda create --name step-seq2seq python=3.7.10
conda activate step-seq2seq
pip install -r requirements.txt 
```

# Run

```
python main.py --model-path model_path --type-model type_model --batch-size batch_size --only-test only_test --max-len max_len
```

`model_path` is the path where you want to save the model, you can to choose between: `models/their_split`, `models/their_split_mixed`, `models/their_split_all_processes`.  

`type_model` is the model of BART, you can to choose between `base` or `large`.

`batch_size` is size of the batch, default is 2

`only_test` is the boolean for only testing the model, default is False

`max_len` used to distinguish between with and without glosses, default is 20 refers to without glosses and 175 is with glosses


for example for training phase:

```
python main.py --model-path models/their_split_all_processes --type-model large --batch-size 2
```

for example for only testing phase:

```
python main.py --model-path models/their_split_all_processes --type-model large --batch-size 2 --only-test True
```
