# ==========================================
# Created by Afshin Khodaveisi (Afshin.khodaveisi@Studio.unibo.it)
# ===========================================

import torch
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from torch.utils.data import  DataLoader
from src.classifier import AMDataset, AMClassifier
import src.util.dataset_util as dataset_util
from src.util.dataset_util import CfgMaper
from sklearn.metrics import classification_report
import logging
import src.util.util as util
import src.util.plot_util as plt_u
import argparse


class Trainer():

    def __init__(self, cfg, task) :
        super(Trainer, self).__init__() 

        # read data
        self.df_sentences = pd.read_pickle(cfg.base_path + "\df_sentences.pkl")
        self.df_annotations = pd.read_pickle(cfg.base_path + "\df_annotations.pkl")

        self.current_task = task
        logger = logging.getLogger("Trainer")
        logger.info("Run ...")
        model = self.build_model(cfg, len(task.labels))
        optimizer = self.build_optimizer(model)
        training_loader, testing_loader = self.build_training_loader(model)

        print("\nTraining is starting ...")
        train_losses = self.trainer(model, optimizer, training_loader)
        print("\nEvaluation is starting ...")
        self.final_output, self.targets = self.test(model, testing_loader)
        print("\n Final report")
        report = self.build_report(self.final_output, self.targets, cfg)
        print(report)

        self.plot_losses(model.epochs, train_losses)


    def get_output(self):
        return dataset_util.build_output(self.final_output, self.targets, self.current_task, self.new_df)

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
        train_loss = 0.0
        for _,data in tqdm(enumerate(training_loader, 0), desc=f"Epoch {epoch + 1} Training"):
            ids = data['ids'].to(model.device, dtype = torch.long)
            mask = data['mask'].to(model.device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(model.device, dtype = torch.long)
            targets = data['targets'].to(model.device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = self.loss_fn(outputs, targets)
            # This code shows only one instance at 500th dataloader instance
            # if _%500==0 and _!= 0:
            #     print(f'    Epoch: {epoch}, Loss:  {loss.item()}')

            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        return train_loss / len(training_loader)

    def trainer(self, model, optimizer, training_loader):
        train_losses = []
        for epoch in range(model.epochs):
            avg_train_loss = self._train(epoch, model, optimizer, training_loader)
            print(f"Epoch {epoch + 1}/{model.epochs} | Training Loss: {avg_train_loss:.4f}")
            train_losses.append(avg_train_loss)
        return train_losses
    

    def build_training_loader(self, model):
   
        # Pre-processing data and data domain
        self.new_df = self.pre_process_data(self.df_sentences, self.df_annotations, self.current_task.name, **model.pre_process_params)

        # Creating the dataset and dataloader 
        train_size = model.train_size
        self.train_data=self.new_df.sample(frac=train_size,random_state=model.random_state)
        self.test_data=self.new_df.drop(self.train_data.index).reset_index(drop=True)
        self.train_data = self.train_data.reset_index(drop=True)

        print("FULL Dataset: {}".format(self.new_df.shape))
        print("TRAIN Dataset: {}".format(self.train_data.shape))
        print("TEST Dataset: {}".format(self.test_data.shape))

        self.training_set = AMDataset(self.train_data, model.tokenizer, model.max_len)
        self.testing_set = AMDataset(self.test_data, model.tokenizer, model.max_len)

        training_loader = DataLoader(self.training_set, **model.train_params)
        testing_loader = DataLoader(self.testing_set, **model.test_params)

        return training_loader,testing_loader
    
    
    def pre_process_data(self, df_sentences, df_annotations, tak_name:str, synonym_fold, deletion_fold) -> pd.DataFrame:
        new_df = pd.DataFrame()
        if tak_name == "AD": 
            new_df['text'] = df_sentences['Text']
            new_df['labels'] = df_sentences['Name'].apply(lambda x: dataset_util.prepare_AD_label(x))
        elif tak_name == "AC":
            new_df['text'] = df_annotations['Text']
            new_df['labels'] = df_annotations['Name'].apply(lambda x: dataset_util.prepare_AC_label(x))
        elif tak_name == "TC":
            df_annotations = dataset_util.preprocessing_data(df_annotations, "Type")
            new_df['text'] = df_annotations['Text']
            new_df['labels'] = df_annotations['Type'].apply(lambda x: dataset_util.prepare_TC_label(x))
        elif tak_name == "SC":
            df_annotations = dataset_util.preprocessing_data(df_annotations, "Scheme")
            new_df['text'] = df_annotations['Text']
            new_df['labels'] = df_annotations['Scheme'].apply(lambda x: dataset_util.prepare_SC_label(x))
        else:
            print("Model Type couldn't be recognised!")

        # Data Augumentation
        new_df = self._augment_data(new_df, synonym_fold, deletion_fold)    
        return new_df

    def _augment_data(self, df, synonym_fold=0, deletion_fold=0):
        augmented_texts = []
        augmented_labels = []
        # Todo: Pipline?
        for _,row in df.iterrows():
            original_text = row['text']
            original_label = row['labels']
            # Deletion Agumntation
            for _ in range(deletion_fold):
                augmented_text = dataset_util.augment_data_deletion(original_text)
                augmented_texts.append(augmented_text)
                augmented_labels.append(original_label)
            # Synonym Agumntation
            for _ in range(synonym_fold):
                augmented_text = dataset_util.augment_data_synonym(original_text)
                augmented_texts.append(augmented_text)
                augmented_labels.append(original_label)



        augmented_df = pd.DataFrame({'text':augmented_texts, 'labels':augmented_labels})
        augmented_df = pd.concat([df, augmented_df], ignore_index=True)

        return augmented_df


    def test(self, model, testing_loader):
        # targets are actual label and output is predicted label
        outputs, targets = self._test(model, testing_loader)

        final_outputs = (np.array(outputs) >= 0.5).astype(int)
        #print("output: " , final_outputs[:10])
        #print("target: ", targets[:10])
        return final_outputs, targets
        
    def _test(self, model, testing_loader):
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
    
    def plot_losses(self, number_epochs, train_losses):
        plt_u.plot_training_loss(number_epochs, train_losses)



def setup(config_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = 42
    print("device is: ", device)

    base_path = os.getcwd()
    configs_folder = os.path.join(os.getcwd(), "src/configs")
    model_path = os.path.join(configs_folder, config_file)




    # tasks = [CfgMaper({'name':'AD', 'labels':['premise','conclusion','neither']}),
    #          CfgMaper({'name':'AC', 'labels':['premise','conclusion']}),
    #          CfgMaper({'name':'TC', 'labels':['L','F']}),
    #          CfgMaper({'name':'SC', 'labels':['Rule', 'Itpr', 'Prec', 'Class', 'Princ', 'Aut']})]
    tasks = [
             CfgMaper({'name':'SC', 'labels':['Rule', 'Itpr', 'Prec', 'Class', 'Princ', 'Aut']})]

    # CfgMaper gets back elements by dot(extended class of dict)
    default_cfg = CfgMaper({'device':device , 'tasks':tasks, 'random_state':random_state ,'base_path':base_path })
                    
    cfg = default_cfg.merge_file(util.load_conf(model_path))
    #print(cfg)
   
    return cfg


def main(config_file):
    cfg = setup(config_file)
    outputs = []
    for task in cfg.tasks:
        print("Start running ",task)
        trainer = Trainer(cfg, task)
        outputs.append(trainer.get_output())
    
    # Save outputs in order to analyze the data
    try:
        dataset_util.save_output_data(outputs, cfg.base_path +'\src\outputs'+ cfg.MODEL.name)
    except RuntimeError as e:
        print(e)

    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and Test")
    # config arg is not mandatory
    #parser.add_argument("--config-file", type=str,  default= "distilbert.yaml",help="Path to the YAML configuration file.")
    parser.add_argument("--config-file", type=str,  required=True, help="Name of YAML configuration file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config_file = args.config_file

    main(config_file)


