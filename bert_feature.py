from transformers import BertTokenizer, BertConfig
from transformers import BertModel
import pandas as pd
import torch
import ipdb
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bert_embedding_features(model_name, seq_list, hidden_layers=2):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model_config = BertConfig.from_pretrained(model_name)
    model_config.output_hidden_states = True
    model_config.output_attentions = True
    model_config.num_hidden_layers = hidden_layers
    print('------------------------layers {}------------------------'.format(hidden_layers))
    model = BertModel.from_pretrained(model_name, config = model_config)
    model.to(device)
    model = model.eval()
    with torch.no_grad():
      encoded_input = tokenizer(seq_list, padding = True, return_tensors='pt').to(device) # batch_encode_plus
      output = model(**encoded_input)
      embedding_features =torch.cat([_.cpu().unsqueeze(0) for _ in output.hidden_states[1:hidden_layers + 1]],0)
      
    return embedding_features

def make_seq(df_data,max_length,model_name,hidden_layers):
  sequence = df_data['Sequence'].tolist()
  sq_=[]
  length = []
  for idx,s in enumerate(sequence):

    if len(s)>=max_length:
      s = s[:max_length]
    # '''
      length.append([len(s)])
      sq_.append(' '.join(s))
      
    else:
      length.append([len(s)])
      sq_.append(' '.join(s))
      sq_[-1] = sq_[-1] + ' [PAD]'*(max_length-len(s))
    
  if len(sq_)<100:
    sq_to_bert = sq_
    raw_embedding_features = bert_embedding_features(model_name, sq_to_bert, hidden_layers=hidden_layers)
    
    embedding_features = raw_embedding_features
  else:
  
    batch_size = 100
    batch = 0
    embedding_features = torch.tensor([])
    while batch < len(sq_):

      sq_to_bert = sq_[batch:min(batch+batch_size,len(sq_))]
      raw_embedding_features = bert_embedding_features(model_name, sq_to_bert, hidden_layers=hidden_layers)
      ef = raw_embedding_features
      batch = min(batch+batch_size,len(sq_))
      embedding_features = torch.cat((embedding_features,ef),1)
  print('=======================================')  
  print('shape of embedidng features:',embedding_features.shape)
  return embedding_features

# embedding_features_dict = [s:make_seq(training_sets[s]) for s in ["Anti-CoV", "Anti-Virus", "non-AVP", "non-AMP"]]

# ipdb.set_trace()
   

if __name__=='__main__':
  training_sets = {
        lab: pd.read_csv("./data/{:s}-train.csv".format(lab))
        for lab in ["Anti-Virus", "non-AVP", "non-AMP",'All-Neg','All-AMP']
      }
  test_sets = {
          lab: pd.read_csv("./data/{:s}-test.csv".format(lab))
          for lab in [ "Anti-Virus", "non-AVP", "non-AMP",'All-Neg','All-AMP']
      }
 
  data_dict = {}
  data_dict['train'] = {}
  data_dict['test'] = {}
  data_dict['train']['datasets'] = training_sets
  data_dict['test']['datasets'] = test_sets


  
  model_name = '/data/public/models/bert-base-uncased'
  max_length = 100
  features_name = 'bert'
  saveroot = "./embedding_features"
  layer = 12
  # def make_features(data_dict, model_name, max_length):
  # layer = 4
  for dtype in ['train','test']:
    for s in [ "Anti-Virus", "non-AVP", "non-AMP",'All-Neg','All-AMP']:
     
      efs = make_seq(data_dict[dtype]['datasets'][s],max_length,model_name,layer)
      print('==========================================================')
      print('{} Done: Length:{}'.format(s,data_dict[dtype]['datasets'][s].shape[0]))

      os.makedirs(os.path.join(saveroot,dtype+'_data'),exist_ok=True)
      torch.save(efs,os.path.join(saveroot,dtype+'_data',f'{s}-{features_name}-{dtype}-{layer}layers.pt'))