# ==========================================
# Created by Afshin Khodaveisi (Afshin.khodaveisi@Studio.unibo.it)
# ===========================================

import torch
from tqdm import tqdm
import numpy as np
from argument_classification import DistilBERTClassifier
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import cuda
from argument_classification import AMDataset
from util import CfgMaper
from sklearn.metrics import classification_report
import logging


class Trainer():

    def __init__(self, cfg) :
        super(Trainer, self).__init__() 

        logger = logging.getLogger("Trainer")
        logger.info("Run ...")
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(model)
        training_loader, testing_loader = self.build_training_loader(model)

        print("\nTraining is starting ...")
        self.trainer(model, optimizer, training_loader)
        print("\nEvaluation is starting ...")
        self.final_output, self.targets = self.test(model, testing_loader)
        print("\n Final report")
        report = self.build_report(self.final_output, self.targets, cfg)
        print(report)

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
            if _%500==0 and _!= 0:
                print(f'    Epoch: {epoch}, Loss:  {loss.item()}')
            
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
        new_df_sentences['labels'] = df_sentences['Name'].apply(lambda x: self.prepare_ad_label(x))

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
    
    def prepare_ad_label(self,label):
        if label == 'prem':
          return [1,0,0]
        elif label == 'conc':
            return [0,1,0] 
        else: 
            return [0,0, 1]
        
    def test(self, model, testing_loader):
        # targets are actual label and output is predicted label
        outputs, targets = self._validation(model, testing_loader)
        final_outputs = (np.array(outputs) >= 0.5).astype(int)
        #final_outputs = temp_outputs.astype(int)
        print("output: " , final_outputs[:10])
        print("target: ", targets[:10])
        return final_outputs, targets
        
    def _validation(self, model, testing_loader):
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in tqdm(enumerate(testing_loader, 0)):
                ids = data['ids'].to(model.device, dtype = torch.long)
                mask = data['mask'].to(model.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(model.device, dtype = torch.long)
                targets = data['targets'].to(model.device, dtype = torch.float)
                outputs = model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets
    
    def build_report(self, predicated_label, actual_label, cfg):
        report = classification_report(actual_label, predicated_label, target_names=cfg.ad_labels)
        return report



def setup():
    #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is: ", device)

    model1 = {"name":"distilbert"}
    model2 = {"name":"distilbert"}
    models = [model1,model2]

    tasks = {"Ac":1 , "AC":2 , "TC":3 , "SC":4}

    max_len = 128
    train_batch_size = 4
    valid_batch_size = 4
    epochs = 1
    learning_rate = 1e-05

    ad_labels = ['premise','conclusion','neither']
    ac_labels = ['premise','conclusion']
    tc_labels = ['L','F']
    sc_labels = ['Rule', 'Itpr', 'Prec', 'Class', 'Princ', 'Aut']

    train_params = {'batch_size': train_batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': valid_batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    return CfgMaper({'device':device , 'max_len':max_len, 'train_batch_size':train_batch_size, 'valid_batch_size':valid_batch_size ,
                     'epochs':epochs, 'learning_rate':learning_rate, 'train_params':train_params, 'test_params':test_params,
                     'ad_labels':ad_labels, 'ac_labels':ac_labels, 'tc_labels':tc_labels, 'sc_labels':sc_labels ,'Model':models ,
                      'Task':tasks })


def main():
    cfg = setup()
    trainer = Trainer(cfg)
    #report_ad = trainer.report()


if __name__ == "__main__":
    main()


