import torch
import torch.nn as nn
import transformers
from transformers import RobertaConfig, RobertaModel

class Model(nn.Module):
    def __init__(self,
                args,
                device = None):
        super(Model, self).__init__()
        self.config = RobertaConfig.from_pretrained(args.from_pretrained)
        self.model = RobertaModel.from_pretrained(args.from_pretrained)
        self.hidden_size = self.config.hidden_size
        self.mlp = torch.nn.Sequential(
                        torch.nn.Linear(self.hidden_size, self.hidden_size),
                        torch.nn.ReLU(),
                        # nn.Dropout(p=args.dropout_prob),
                        torch.nn.Linear(self.hidden_size, self.hidden_size),
                        )
        #self.extra_token_embeddings = nn.Embedding(args.new_tokens, self.model.config.hidden_size)

    def forward(self, input_ids, attention_mask, prompt_label_idx, labels):
        batch_size = input_ids.shape[0]
        outputs = self.model(input_ids, attention_mask = attention_mask)
        encodes = outputs[0]

        #print('batch_size:',batch_size)
        #print('prompt_idx', prompt_label_idx)
        #print('labels', labels)
        #print(self.model.embeddings.word_embeddings.weight[1099].shape)

        logits = [torch.mm(encodes[i, prompt_label_idx[i]].unsqueeze(0), self.model.embeddings.word_embeddings.weight.transpose(1,0)) for i in range(batch_size)]
        #logits = [self.model.embeddings.word_embeddings.weight for i in range(batch_size)]
        return logits
