import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import numpy as np
from torch import LongTensor,tensor


class DataSet():
  def __init__(self,dataframe,tokenizer):
    
    data_x = list(dataframe['text'].values)
    self.labels = LongTensor(dataframe['label'].values)
    self.encodings = tokenizer(data_x,max_length = 300,truncation=True, padding=True)

  def __getitem__(self,idx):
    
    item = {key:tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = self.labels[idx]

    return (item)

  def __len__(self):
    return len(self.labels)


class Tokenizer:
    def __init__(self,tokenizer_path=None):
        """If tokenizer_path is none and tokenizer name is None, 
           we directly download the tokenizer from t5-small
           tokenizer
        """
        self.tokenizer_path = tokenizer_path
        
        if tokenizer_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


    def save_tokenizer(self,path):
        self.tokenizer.save_pretrained(path)


class DistilBertModelForClassification(Tokenizer):
    def __init__(self,label_words,num_labels=None,model_path=None,tokenizer_path = None):
        super().__init__(tokenizer_path = tokenizer_path)
        self.device = 'cpu'
        self.index_to_word = dict({id:val for id,val in enumerate(label_words)})
        
        if model_path is None:
            self.model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=num_labels)
        else:
            print("Loading Model from path {path}".format(path=model_path))

            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
    
    def set_trainer(self,dataset,shuffle:bool=True,batch_size:int=32,epoch:int=10,lr:float=1e-4):
        dataset = DataSet(dataset,self.tokenizer)
        self.train_loader = DataLoader(dataset,shuffle=shuffle,batch_size=batch_size)
        num_training_steps=epoch * len(self.train_loader)
        self.num_epochs = epoch
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.lr_scheduler = get_scheduler(
        name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        self.progress_bar = tqdm(range(num_training_steps))

    def train(self):

        self.model = self.model.to(self.device)
        self.model.train()
        
        for i in range(self.num_epochs):
            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out.loss
                loss.backward()
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                self.progress_bar.update(1)

            print("Training Loss: ",loss.item())

    def save_model_to(self,model_path):
        self.model.save_pretrained(model_path)

    def load_model_from(self,model_path):
        return AutoModelForSequenceClassification.from_pretrained(model_path)

    def get_evaluations(self,dataset,shuffle:bool=True,batch_size:int=8):

        self.model = self.model.to(self.device)
        dataset = DataSet(dataset,self.tokenizer)
        data_loader = DataLoader(dataset,shuffle=shuffle,batch_size=batch_size)
        return self.evaluate(data_loader)


    def evaluate(self,loader):
        outputs = []
        accs = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch).logits

            output = torch.argmax(out,dim=1)
            outputs.append(output.cpu().data.numpy())
    
            acc = accuracy_score(list(output.cpu().data.numpy()),list(batch['labels'].cpu().data.numpy()))

            accs.append(acc)

        mean_acc = np.mean(accs)

        return mean_acc,accs

    def set_device(self,device:str='cpu'):
        self.device = device
 

    def infer(self,text,beam_width:int=1):

        self.model = self.model.to(self.device)
        
        encod_text = self.tokenizer(text,max_length = 300,truncation=True, padding=True,return_tensors='pt')
        encod_text = {k:v.to(self.device) for k,v in encod_text.items()}
        
        out = self.model(**encod_text).logits
        if beam_width>1:
            raise Exception("sorry, beam width decoding not implemented yet")
        else:
            out = torch.argmax(out,dim=1).cpu().data.numpy()
            pred = self.index_to_word[out[0]]

            return pred