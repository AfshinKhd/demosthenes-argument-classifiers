# ==========================================
# Created by Afshin Khodaveisi (Afshin.khodaveisi@Studio.unibo.it)
# ===========================================

import torch
from tqdm import tqdm
from argument_classification import DistilBERTClassifier
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import cuda
from argument_classification import AMDataset
from util import CfgMaper


class Trainer():

    def __init__(self, cfg) :
        super(Trainer, self).__init__() 

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(model)
        training_loader, testing_loader = self.build_training_loader(model)

        trainer = self.trainer(model, optimizer, training_loader)

    def build_model(self, cfg):
        model = DistilBERTClassifier(cfg)
        model.to(model.device)
        print(model)
        return model
    
    def build_optimizer(self, model):
        return torch.optim.Adam(params =  model.parameters(), lr=model.learning_rate)
    
    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)
    
    def _train(self, epoch, model, optimizer, training_loader):
        model.train()
        for _,data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(model.device, dtype = torch.long)
            mask = data['mask'].to(model.device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(model.device, dtype = torch.long)
            targets = data['targets'].to(model.device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = self.loss_fn(outputs, targets)
            if _%5000==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            
            loss.backward()
            optimizer.step()

    def trainer(self, model, optimizer, training_loader):
        for epoch in range(model.epochs):
            self._train(epoch, model, optimizer, training_loader)

    def build_training_loader(self, model):
        # read data
        base_path = os.getcwd() 
        df_sentences = pd.read_pickle(base_path + "\df_sentences.pkl")
        df_annotations = pd.read_pickle(base_path + "\df_annotations.pkl")

        # Pre-processing data and data domain
        new_df_sentences = pd.DataFrame()
        new_df_sentences['text'] = df_sentences['Text']
        new_df_sentences['labels'] = df_sentences['Name'].apply(lambda x: self.prepare_ac_label(x))

        # Creating the dataset and dataloader 
        train_size = 0.8
        self.train_data=new_df_sentences.sample(frac=train_size,random_state=200)
        self.test_data=new_df_sentences.drop(self.train_data.index).reset_index(drop=True)
        self.train_data = self.train_data.reset_index(drop=True)

        print("FULL Dataset: {}".format(new_df_sentences.shape))
        print("TRAIN Dataset: {}".format(self.train_data.shape))
        print("TEST Dataset: {}".format(self.test_data.shape))

        self.training_set = AMDataset(self.train_data, model.tokenizer, model.max_len)
        self.testing_set = AMDataset(self.test_data, model.tokenizer, model.max_len)

        training_loader = DataLoader(self.training_set, **model.train_params)
        testing_loader = DataLoader(self.testing_set, **model.test_params)

        return training_loader,testing_loader
    
    def prepare_ac_label(self,label):
        if label == 'prem':
          return [1,0,0]
        elif label == 'conc':
            return [0,1,0] 
        else: 
            return [0,0, 1]


def setup():
    #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is: ", device)

    max_len = 128
    train_batch_size = 4
    valid_batch_size = 4
    epochs = 1
    learning_rate = 1e-05

    train_params = {'batch_size': train_batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': valid_batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    return CfgMaper({'device':device , 'max_len':max_len, 'train_batch_size':train_batch_size, 'valid_batch_size':valid_batch_size ,
                     'epochs':epochs, 'learning_rate':learning_rate, 'train_params':train_params, 'test_params':test_params })


def main():
    print("here")
    cfg = setup()
    trainer = Trainer(cfg)
    trainer.build_model()


if __name__ == "__main__":
    main()


