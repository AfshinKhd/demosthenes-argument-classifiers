# ==========================================
# Created by Afshin Khodaveisi (Afshin.khodaveisi@Studio.unibo.it)
# ===========================================

import torch
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from torch.utils.data import  DataLoader
from classifier import AMDataset, AMClassifier
import util.dataset_util as dataset_util
from util.dataset_util import CfgMaper
from sklearn.metrics import classification_report
import logging
import util.util as util


class Trainer():

    def __init__(self, cfg, task) :
        super(Trainer, self).__init__() 

        self.current_task = task
        logger = logging.getLogger("Trainer")
        logger.info("Run ...")
        model = self.build_model(cfg, len(task.labels))
        optimizer = self.build_optimizer(model)
        training_loader, testing_loader = self.build_training_loader(model)

        print("\nTraining is starting ...")
        self.trainer(model, optimizer, training_loader)
        print("\nEvaluation is starting ...")
        self.final_output, self.targets = self.test(model, testing_loader)
        print("\n Final report")
        report = self.build_report(self.final_output, self.targets, cfg)
        print(report)

    def build_model(self, cfg, num_outputs):

        model = AMClassifier(cfg, num_outputs)

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
        if self.current_task.name == "AD":
            df_sentences = pd.read_pickle(base_path + "\df_sentences.pkl")
        else:
            df_annotations = pd.read_pickle(base_path + "\df_annotations.pkl")
        

        # Pre-processing data and data domain
        new_df = pd.DataFrame()
        if self.current_task.name == "AD": 
            new_df['text'] = df_sentences['Text']
            new_df['labels'] = df_sentences['Name'].apply(lambda x: dataset_util.prepare_AD_label(x))
        elif self.current_task.name == "AC":
            new_df['text'] = df_annotations['Text']
            new_df['labels'] = df_annotations['Name'].apply(lambda x: dataset_util.prepare_AC_label(x))
        elif self.current_task.name == "TC":
            df_annotations = dataset_util.preprocessing_data(df_annotations, "Type")
            new_df['text'] = df_annotations['Text']
            new_df['labels'] = df_annotations['Type'].apply(lambda x: dataset_util.prepare_TC_label(x))
        elif self.current_task.name == "SC":
            df_annotations = dataset_util.preprocessing_data(df_annotations, "Scheme")
            new_df['text'] = df_annotations['Text']
            new_df['labels'] = df_annotations['Scheme'].apply(lambda x: dataset_util.prepare_SC_label(x))
        else:
            print("Model Type couldn't be recognised!")
        

        # Creating the dataset and dataloader 
        train_size = model.train_size
        self.train_data=new_df.sample(frac=train_size,random_state=model.random_state)
        self.test_data=new_df.drop(self.train_data.index).reset_index(drop=True)
        self.train_data = self.train_data.reset_index(drop=True)

        print("FULL Dataset: {}".format(new_df.shape))
        print("TRAIN Dataset: {}".format(self.train_data.shape))
        print("TEST Dataset: {}".format(self.test_data.shape))


        # result = self.test_data['labels'].apply(lambda x: x == [1,0,0,0,0,0])

        # print(result.sum())

        self.training_set = AMDataset(self.train_data, model.tokenizer, model.max_len)
        self.testing_set = AMDataset(self.test_data, model.tokenizer, model.max_len)

        training_loader = DataLoader(self.training_set, **model.train_params)
        testing_loader = DataLoader(self.testing_set, **model.test_params)

        return training_loader,testing_loader
    

        
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
        report = classification_report(actual_label, predicated_label, target_names=self.current_task.labels)
        return report



def setup():
    #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = 42
    print("device is: ", device)
    print("\n")

    base_path = os.getcwd()
    configs_folder = os.path.join(os.getcwd(), "src/configs")
    model_path = os.path.join(configs_folder, "deberta.yaml")

    tasks = [CfgMaper({'name':'AD', 'labels':['premise','conclusion','neither']}),
             CfgMaper({'name':'AC', 'labels':['premise','conclusion']}),
             CfgMaper({'name':'TC', 'labels':['L','F']}),
             CfgMaper({'name':'SC', 'labels':['Rule', 'Itpr', 'Prec', 'Class', 'Princ', 'Aut']})]

    # CfgMaper gets back elements by dot(extended class of dict)
    default_cfg = CfgMaper({'device':device , 'tasks':tasks, 'random_state':random_state ,'base_path':base_path })
                    
    
    cfg = default_cfg.merge_file(util.load_conf(model_path))
   
    return cfg

def scheduler(cfg):
    
    for task in cfg.tasks:
        print("Start running ",task)
        trainer = Trainer(cfg, task)

    # print("Start running AD")
    # cfg.MODEL = CfgMaper({"name":"distilbert" , "num_output":3, "task":"AD", "task_label":cfg.ad_labels})
    # print("cfg: ", cfg.MODEL.name)
    # trainer = Trainer(cfg)

    # print("\nStart running AC")
    # cfg.MODEL = CfgMaper({"name":"distilbert" , "num_output":2, "task":"AC", "task_label":cfg.ac_labels})
    # print("cfg: ", cfg.MODEL)
    # trainer = Trainer(cfg)

    # print("\nStart running AC")
    # cfg.MODEL = CfgMaper({"name":"distilbert" , "num_output":2, "task":"TC", "task_label":cfg.tc_labels})
    # print("cfg: ", cfg.MODEL)
    # trainer = Trainer(cfg)

    # print("\nStart running AC")
    # cfg.MODEL = CfgMaper({"name":"distilbert" , "num_output":6, "task":"SC", "task_label":cfg.sc_labels})
    # print("cfg: ", cfg.MODEL)
    # trainer = Trainer(cfg)



def main():
    cfg = setup()
    scheduler(cfg)
    
    #report_ad = trainer.report()


if __name__ == "__main__":
    main()


