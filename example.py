'''
from distilbert import *
import pandas as pd


#                          INFERENCE

labels = 3
words = ['Neutral','Positive','Negetive']
model_path = 'path/to/model
tokenizer_path = 'path/to/tokenizer' (if available, else it'll download from huggingface hub)

model = DistilBertModelForClassification(num_labels=labels,label_words=words,model_path=model_path,tokenizer_path=tokenizer)

example_text = "this model is good"

model.infer(example_text)  # output: Positive

#                          TRAINING

path_to_csv = "path/to/csv"
df = pd.read_csv(path_to_csv)
df = df[['text','label']]

#just pass the csv file
model.set_trainer(new_df,batch_size=32,epoch=30)

#set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.set_device(device)

#train model
model.train()

#save model
path_to_save = path/to/save_model
model.save_model_to(path/to/save_model)

#load model
#you can load model by given the path to the folder of the model 
#before initialising the DistilBertModelForSequenceClassification like this

model = DistilBertModelForClassification(label_words=words,model_path=path_to_save)


'''