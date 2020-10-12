import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import transformers
from transformers import AutoModel, AutoTokenizer, AdamW

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

df_test = pd.read_csv("test2.csv", sep='\t')
df_train = pd.read_csv("train2.csv", sep='\t')
df_val=pd.read_csv("val2.csv", sep='\t')

df_train.head()

from ipywidgets import IntProgress

bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

train_text = df_train['text']
train_labels= df_train['grammar']
train_number = df_train['number']
#train_text max len
val_text = df_val['text']
val_labels=df_val['grammar']

def mapLabels(x):
    return {
        'Nom': 0,
        'Acc': 1,
        'Sub': 2,
        'Ine': 3,
        'Sup': 4,
        'Ins': 5
    }.get(x, 9) 

train_label_list = list(set(train_labels))

train_numbered_labels = []
for label in train_labels:
    train_numbered_labels.append(mapLabels(label))
    
val_label_list = list(set(val_labels))	
val_numbered_labels = []
for label in val_labels:
    val_numbered_labels.append(mapLabels(label))

tokens_train = tokenizer(train_text.values.tolist(), max_length=512, padding=True, truncation=True, return_tensors="pt")


tokens_val = tokens_train = tokenizer(val_text.values.tolist(), max_length=512, padding=True, truncation=True, return_tensors="pt")


train_seq = torch.as_tensor(tokens_train['input_ids'])
train_mask = torch.as_tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_numbered_labels)

val_seq = torch.as_tensor(tokens_val['input_ids'])
val_mask = torch.as_tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_numbered_labels)

batch_size = 32

train_data = TensorDataset(train_seq, train_mask, train_y)

train_= RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)

val_sampler = SequentialSampler(val_data)

val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

class myBert(nn.Module):

    def __init__(self, bert):

        super(myBert, self).__init__()

        self.bert = bert
        self.dropuot=nn.Dropout(0.1)
        self.relu=nn.ReLU()
        self.fc1 = nn.Linear(768,2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        x=self.fc1(cls_hs)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.softmax(x)
        return x

model = myBert(bert)

optimizer = AdamW(model.parameters(),
                  lr = 1e-5) 

cross_entropy = nn.NLLLoss()

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


def eval():

    model.eval()
    
    total_loss, total_accuracy = 0, 0
    total_preds = []
    
    for step,batch in enumerate(val_dataloader):
         # progress update after every 5 batches.
        if step % 5 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
        sent_id, mask, labels = batch

        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds,labels)
            total_loss = total_loss + loss.item()
            total_preds.append(preds)


        avg_loss = total_loss/len(val_dataloader)
        total_preds  = np.concatenate(total_preds, axis=0)

    return val_loss, total_preds

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

epochs = 10

#for each epoch
for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _ = train()
    
    #evaluate model
    valid_loss, _ = evaluate()
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')
	
	
with torch.no_grad():
  preds = model(test_seq.to(device), test_mask.to(device))
  preds = preds.detach().cpu().numpy()
	
preds = np.argmax(preds, axis = 1)
