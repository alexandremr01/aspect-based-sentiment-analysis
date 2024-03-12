from typing import List

import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm


POLARITY_MAP = {'positive': 2, 'negative': 0, 'neutral': 1}

CATEGORY_MAP = {
    'AMBIENCE#GENERAL': 'ambiance of',  
    'DRINKS#PRICES': 'prices of drink',  
    'DRINKS#QUALITY': 'quality of drink',  
    'DRINKS#STYLE_OPTIONS': 'style options of drink',  
    'FOOD#PRICES': 'price of food',  
    'FOOD#QUALITY': 'quality of food',  
    'FOOD#STYLE_OPTIONS': 'style options of food',  
    'LOCATION#GENERAL': 'location of',  
    'RESTAURANT#GENERAL': 'restaurant',  
    'RESTAURANT#MISCELLANEOUS': 'restaurant',  
    'RESTAURANT#PRICES': 'price of restaurante',  
    'SERVICE#GENERAL': 'service of',  
}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_dataframe(filename):
    df = pd.read_csv(filename, sep='	', header=None)
    df.columns = ['sentiment', 'category', 'target', 'offset', 'sentence']
    df['sentiment_label'] = df.sentiment.map(POLARITY_MAP)
    df['parsed_category'] = df.category.map(CATEGORY_MAP)
    df['compound_target'] = df.apply(lambda row : row.parsed_category + ' ' + row.target, axis=1)
    return df


class ABSADistilBert(torch.nn.Module):
    def __init__(self, dropout_rate, weights=None):
        super(ABSADistilBert, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 256)
        self.linear2 = torch.nn.Linear(256, 32)
        self.linear3 = torch.nn.Linear(32, 3)
        self.relu = torch.nn.ReLU()
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, ids, masks=None):
        out = self.bert(
            input_ids=ids,
            attention_mask=masks,
          )
        out = out['last_hidden_state'][:, 0, :]
        out = self.dropout(out)
        out = self.linear(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        return self.linear3(out)
    

class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
     """
    
    def __init__(self):
        """
        This should create and initilize the model. Does not take any arguments.
        
        """
        self.model = ABSADistilBert(dropout_rate=0, weights=None)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    
    def get_dataloader(self, df, shuffle, batch_size):
        tokenized = self.tokenizer(df.sentence.tolist(), df.compound_target.tolist(), padding=True)
        dataset = Dataset(tokenized, df.sentiment_label)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader
    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        # Hyperparameters
        lr = 1.2e-5
        batch_size = 64
        weight_decay_factor = 0.048
        epochs = 5

        df = get_dataframe(train_filename)
        train_dataloader = self.get_dataloader(df, shuffle=False, batch_size=batch_size)

        self.model.to(device)
        weight_decay = weight_decay_factor * np.sqrt(batch_size/(df.shape[0]*epochs))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            epoch_loss = 0
            labels = []
            y_pred = []
            self.model.train()
            for batch in tqdm(train_dataloader):
                output = self.model(
                    ids=batch['input_ids'].to(device),
                    masks=batch['attention_mask'].to(device),
                )
                _, predictions = torch.max(output, dim=1)
                loss = self.model.loss_fn(output, batch['labels'].to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                labels.append(batch['labels'])
                y_pred.append(predictions.cpu())

            labels = torch.cat(labels)
            y_pred = torch.cat(y_pred)
            train_acc = accuracy_score(labels, y_pred)
            microf1 = f1_score(labels, y_pred, average='micro')
            epoch_loss /= len(train_dataloader)
            
            print("\nepoch: {}\tloss: {}\ttrain f1:{}".format(epoch, epoch_loss, microf1))

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        self.model.to(device)

        df = get_dataframe(data_filename)
        dataloader = self.get_dataloader(df, shuffle=False, batch_size=64)
        classes = np.array(['negative', 'neutral', 'positive'])

        y_pred = []
        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(dataloader):
                output = self.model(
                    ids=batch['input_ids'].to(device),
                    masks=batch['attention_mask'].to(device),
                )
                _, predictions = torch.max(output, dim=1)
                y_pred.append(predictions.cpu())

        y_pred = torch.cat(y_pred)
        return list(classes[y_pred])
