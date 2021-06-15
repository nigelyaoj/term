import torch.nn.functional as F
from torch import nn
from transformers import BertTokenizer
from transformers import BertModel


class BertLstm(nn.Module):

    def __init__(self):
        super(BertLstm, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.model = BertModel.from_pretrained("bert-base-cased",output_hidden_states = True,)

        
        