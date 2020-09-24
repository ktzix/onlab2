import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import transformers
from transformers import AutoModel, AutoTokenizer, AdamW

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

df_test = pd.read("")
df_train = pd.read("")
df_val=pd.read("")

bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

train_text = df_train('text')
train_y = df_train('label')
#train_text max len


#tokens_train = tokenizer.tokenize(trainlist?)
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())


batch_size = 32

train_data = TensorDataset(train_seq, train_mask, train_y)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

class myBert(nn.Module):

	def __init__(self, bert):
	
		super(myBert, self).__init__()

		self.bert = bert
		self.dropuot=nn.Dropout(0.1)
		self.relu=nn.ReLU()
		self.fc1 = nn.Linear(768,2)
		self.softmax = nn.LogSoftMax(dim=1)

	def forward(self, sent_id, mask)
		_, cls_hs = self.bert(sent_id, attention_mask=mask)
		x=self.fc1(cls_hs)
		x=self.relu(x)
		x=self.dropout(x)
		x=self.softmax(x)
		return x


model = myBert(bert)

optimizer = AdamW(model.parameters(),
                  lr = 1e-5) 

def train():
  
  model.train()

  total_loss, total_accuracy = 0, 0
  
  total_preds=[]
  
  for step,batch in enumerate(train_dataloader):
    
    # progress update after every 5 batches.
    if step % 5 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    batch = [r.to(device) for r in batch]
 
    sent_id, mask, labels = batch

    model.zero_grad()        

    preds = model(sent_id, mask)

    loss = cross_entropy(preds, labels)

    total_loss = total_loss + loss.item()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    preds=preds.detach().cpu().numpy()

    total_preds.append(preds)

  avg_loss = total_loss / len(train_dataloader)
  
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds
