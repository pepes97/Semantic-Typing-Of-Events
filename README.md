# Semantic-Typing-Of-Events


# Setup

```
conda create --n semantic-typing-sveva python=3.7.10
conda activate semantic-typing-sveva
pip install -r requirements.txt
```

# Run

```
python main.py --model-path model_path --type-model type_model --batch-size batch_size --only-test only_test --max-len max_len
```

`model_path` is the path where you want to save the model, you can to choose between: `models/their_split`, `models/their_split_mixed`, `models/their_split_all_processes`, `models/their_split_WORD`.  

`type_model` is the model of BART, you can to choose between `base` or `large`.

`batch_size` is size of the batch, default is 4

`only_test` is the boolean for only testing the model

`max_len` is max length of sentences in generation phase