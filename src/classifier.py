import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel , AutoTokenizer, XLMRobertaModel, DebertaV2Model
import logging
logging.basicConfig(level=logging.ERROR)





class AMClassifier(torch.nn.Module):

    def __init__(self, cfg, num_outputs) :
        super(AMClassifier, self).__init__()
    
        self.device = cfg.device
        self.random_state = cfg.random_state
        self.train_params= cfg.train_params
        self.test_params = cfg.test_params
        self.epochs = cfg.MODEL.epochs

        # Sections of config
        self.train_size = cfg.MODEL.train_size
        self.max_len = cfg.MODEL.max_len
        self.train_batch_size = cfg.MODEL.train_batch_size
        self.valid_batch_size = cfg.MODEL.valid_batch_size
        self.learning_rate = cfg.MODEL.learning_rate
        self.num_pre_output = cfg.MODEL.num_pre_output
        
        if cfg.MODEL.name == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained(cfg.MODEL.model_id, truncation=True, do_lower_case=True)
            self.l1 = DistilBertModel.from_pretrained(cfg.MODEL.model_id)
        elif cfg.MODEL.name == 'deberta':
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.model_id, truncation=True, do_lower_case=True, use_fast= False)
            self.l1 = DebertaV2Model.from_pretrained(cfg.MODEL.model_id)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.model_id, truncation=True, do_lower_case=True)
            self.l1 = XLMRobertaModel.from_pretrained(cfg.MODEL.model_id)           
        # elif  cfg.MODEL.name == 'xlm-r':
        #     self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.model_id, truncation=True, do_lower_case=True)
        #     self.l1 = XLMRobertaModel.from_pretrained(cfg.MODEL.model_id)
        # else:
        #     print("The model couldn't be recognised!")
        

        # Creating customized model 
        self.pre_classifier = torch.nn.Linear(self.num_pre_output, self.num_pre_output)
        self.dropout = torch.nn.Dropout(0.1)
        self.ac_classifier = torch.nn.Linear(self.num_pre_output, num_outputs)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.ac_classifier(pooler)
        return output
    


class AMDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    