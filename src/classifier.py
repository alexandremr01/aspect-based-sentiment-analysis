from typing import List

import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm

from copy import deepcopy

POLARITY_MAP = {'positive': 2, 'negative': 0, 'neutral': 1}

CATEGORY_MAP = {
    'AMBIENCE#GENERAL': 'ambiance',  
    'DRINKS#PRICES': 'prices of drink',  
    'DRINKS#QUALITY': 'quality of drink',  
    'DRINKS#STYLE_OPTIONS': 'style options of drink',  
    'FOOD#PRICES': 'price of food',  
    'FOOD#QUALITY': 'quality of food',  
    'FOOD#STYLE_OPTIONS': 'style options of food',  
    'LOCATION#GENERAL': 'location',  
    'RESTAURANT#GENERAL': 'restaurant',  
    'RESTAURANT#MISCELLANEOUS': 'restaurant',  
    'RESTAURANT#PRICES': 'price of restaurante',  
    'SERVICE#GENERAL': 'service',  
}

category_ix = {k: i for i, k in enumerate(CATEGORY_MAP.keys())}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, categories, target_ids):
        self.encodings = encodings
        self.labels = labels
        self.categories = categories
        self.target_ids = target_ids

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['category'] = torch.tensor(self.categories[idx])
        item['target_id'] = torch.tensor(self.target_ids[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_dataframe(filename, has_labels=True):
    df = pd.read_csv(filename, sep='	', header=None)
    columns = ['sentiment', 'category', 'target', 'offset', 'sentence']

    df.columns = columns
    
    df['parsed_category'] = df.category.map(CATEGORY_MAP)
    df['compound_target'] = df.apply(lambda row : row.parsed_category + ' ' + row.target, axis=1)
    df['beggining'] = df.offset.str.split(':').str[0].astype(int)
    df['end'] = df.offset.str.split(':').str[1].astype(int)
    df['category_ix'] = df.category.map(category_ix)
    df['compound_target'] = df.apply(lambda row : row.parsed_category + ' ' + row.target, axis=1)
    df['sentiment_label'] = df.sentiment.map(POLARITY_MAP)

    return df


class ABSADistilBert(torch.nn.Module):
    def __init__(self, dropout_rate, n_categories, sep_token_id, weights=None):
        super(ABSADistilBert, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.models = []
        gen_layers = []
        self.n_categories = n_categories
        for _ in range(n_categories):
          layers = [
              torch.nn.Linear(2*self.bert.config.hidden_size, 256),
              torch.nn.Linear(256, 32),
              torch.nn.Linear(32, 3)
          ]
          self.models.append(layers)
          gen_layers.extend(layers)
        self.linears = torch.nn.ModuleList(gen_layers)
        self.relu = torch.nn.ReLU()
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.sep_token_id = sep_token_id

    def forward(self, ids, target_ids, category_ids, masks=None):
        batch_size = ids.size(0)

        bert_out = self.bert(
            input_ids=ids,
            attention_mask=masks,
          )
        sep_ix = torch.argwhere(ids == self.sep_token_id)
        #first has to consider one more SEP token than categories (last one)
        out = bert_out['last_hidden_state'][sep_ix[:, 0], sep_ix[:, 1], :].reshape(batch_size, self.n_categories+1, -1)
        #then discard last SEP
        out = out[:, :self.n_categories, :]
        out = self.dropout(out)

        out_target = bert_out['last_hidden_state'][range(batch_size), target_ids, :]
        out_target = self.dropout(out_target)
        full_out = torch.zeros((batch_size, self.n_categories, 3)).cuda()
        for cat in range(self.n_categories):
          out_cat = out[:, cat, :]
          out_cat = torch.cat([out_target, out_cat], dim=1)
          out_cat = self.models[cat][0](out_cat)
          out_cat = self.relu(out_cat)
          out_cat = self.dropout(out_cat)
          out_cat = self.models[cat][1](out_cat)
          out_cat = self.relu(out_cat)
          out_cat = self.dropout(out_cat)
          out_cat = self.models[cat][2](out_cat)
          full_out[:, cat, :] = out_cat

        output = full_out[range(batch_size), category_ids, :]

        return output

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
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.model = ABSADistilBert(dropout_rate=0.2, 
                                    sep_token_id=self.tokenizer.sep_token_id, 
                                    weights=None,
                                    n_categories=len(CATEGORY_MAP))

        cat = list(CATEGORY_MAP.values())
        self.pair_sentence = cat[0]
        for x in cat[1:]:
            self.pair_sentence += '[SEP]' + x
    
    
    def get_dataloader(self, df, shuffle, batch_size):
        tokenized = self.tokenizer(df.sentence.tolist(), [self.pair_sentence]* len(df), padding=True)
        target_id = df.apply(lambda row : tokenized.char_to_token(batch_or_char_index=row.name, char_index=row.beggining, sequence_index=0), axis=1)
        dataset = Dataset(tokenized, df.sentiment_label, df.category_ix, target_id)
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
        lr = 5e-5
        batch_size = 64
        weight_decay_factor = 0.048
        epochs = 15

        df = get_dataframe(train_filename)
        train_dataloader = self.get_dataloader(df, shuffle=True, batch_size=batch_size)

        # We use dev accuracy to select the best model
        df_dev = get_dataframe(dev_filename)
        dev_dataloader = self.get_dataloader(df_dev, shuffle=False, batch_size=batch_size)

        self.model.to(device)
        weight_decay = weight_decay_factor * np.sqrt(batch_size/(df.shape[0]*epochs))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_dev_result, checkpoint = None, None

        for epoch in range(epochs):
            epoch_loss = 0
            labels = []
            y_pred = []
            self.model.train()
            for batch in tqdm(train_dataloader):
                output = self.model(
                    ids=batch['input_ids'].to(device),
                    masks=batch['attention_mask'].to(device),
                    target_ids=batch['target_id'].to(device),
                    category_ids=batch['category']
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
            
            y_test_hat = self._predict(dev_dataloader, device)
            dev_f1 = f1_score(df_dev.sentiment, y_test_hat, average='micro')
            if best_dev_result is None or dev_f1 > best_dev_result:
                best_dev_result = dev_f1
                checkpoint = deepcopy(self.model.state_dict())
            print("\nepoch: {}\tloss: {}\ttrain f1:{}\ttest f1:{}".format(epoch, epoch_loss, microf1, dev_f1))
        self.model.load_state_dict(checkpoint)

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
        return self._predict(dataloader, device)
    
    def _predict(self, dataloader, device):
        classes = np.array(['negative', 'neutral', 'positive'])

        y_pred = []
        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(dataloader):
                output = self.model(
                    ids=batch['input_ids'].to(device),
                    masks=batch['attention_mask'].to(device),
                    target_ids=batch['target_id'].to(device),
                    category_ids=batch['category']
                )
                _, predictions = torch.max(output, dim=1)
                y_pred.append(predictions.cpu())

        y_pred = torch.cat(y_pred)
        y_hat = list(classes[y_pred])
        return y_hat
