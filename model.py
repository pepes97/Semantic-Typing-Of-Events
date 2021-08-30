import torch.nn as nn

class HyperparamsBART():
   def __init__(self):
     self.freeze_encoders = False
     self.freeze_embeddings = False 

class Seq2SeqModelBART(nn.Module):
  def __init__(self, tokenizer, model, hparams):
    super().__init__()
    self.tokenizer = tokenizer
    self.model = model

    if hparams.freeze_encoders:
      self.freeze_params(self.model.get_encoder())

    if hparams.freeze_embeddings:
      self.freeze_embeds

  def forward(self, input_ids, **kwargs):
    return self.model(input_ids, **kwargs)
  
  def freeze_params(self, model):
    ''' Function that takes a model as input (or part of a model) and freezes 
    the layers for faster training'''
    for layer in model.parameters():
      layer.requires_grade = False

  def freeze_embeds(self):
    ''' freeze the positional embedding parameters of the model '''
    self.freeze_params(self.model.model.shared)
    for d in [self.model.model.encoder, self.model.model.decoder]:
      self.freeze_params(d.embed_positions)
      self.freeze_params(d.embed_tokens)
      
  def shift_tokens_right(self, input_ids, pad_token_id):
    """ Shift input ids one token to the right, 
        and wrap the last non pad token (usually <eos>).
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens