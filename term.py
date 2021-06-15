import torch
import pandas as pd
from transformers import BertTokenizer
from transformers import BertModel
from tqdm import tqdm
import pickle

model = BertModel.from_pretrained("bert-base-cased",output_hidden_states = True,)

term_dict = pickle.load(open("term_input_ids.pt", "rb"))

input_ids = term_dict["id"]
can_dataset_pos = term_dict["can_dataset_pos"]
attention_masks = term_dict["attention_masks"]
labels = term_dict["label"]

print("loaded")


model.eval()
model.to("cuda:0")
candicate_embeddings = []
n = len(input_ids)
with torch.no_grad():  # 将输入传入模型，得到每一层的输出信息，这里的encoded_layers为12层，可以打印验证
    for i in range(n):
        outputs = model(input_ids[i].to("cuda:0"))
        hidden_states = outputs[2]
        
        token_embeddings = torch.cat(hidden_states, dim=0) # 13 * sen_len * 768
        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1, 0, 2)# sen_len * 13 * 768
        # For each token in the sentence...
        token_vecs_sum = torch.sum(token_embeddings[:,-4:,:], dim=1)
        start, end = can_dataset_pos[i]
        candicate_embeddings.append(token_vecs_sum[start:end].cpu())

        if i % 5000 == 0:
            print(i)


term2_dict = {"can_embeddings": candicate_embeddings,
              "labels": labels}

pickle.dump(term2_dict, open("can_embeddings.pt", "wb"))
print("done!")