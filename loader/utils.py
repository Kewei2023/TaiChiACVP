from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from Bio import SeqIO
import pandas as pd
import numpy as np
import ipdb
import torch
import time
import datetime
import random
import os
from tqdm import tqdm
from .dataset import LSDataset
from .preprocess_data import clear_binary_data
from ..utils.std_logger import Logger
import re



def collate_fn(batch):
    '''
    sequence,name2id, label_dict
    '''
    seqs = [_[0] for _ in batch]
    name2id_list = [_[1] for _ in batch]
    label_dicts_list = [_[2] for _ in batch]

    tmp_dict = defaultdict(list)
    name2id = {}
    
    for d_ in name2id_list:
        for k, v in d_.items():
            tmp_dict[k].append(v)

    for k, v in tmp_dict.items():
        name2id[k] = v

    tmp_dict = defaultdict(list)
    label_dict = {}
    
    for d_ in label_dicts_list:
        for k, v in d_.items():
            tmp_dict[k].append(v)

    for k, v in tmp_dict.items():
        label_dict[k] = v

    return seqs, name2id, label_dict



def load_raw_seq(dataset_file,cfg):

  # all_data = read_fasta(dataset_file,cfg)
  all_data = pd.read_csv(dataset_file)
  all_data = all_data.sample(frac=1).reset_index(drop=True)

  if cfg.task.type == "binary":
    sequence, seqName, seqID, label_dict = clear_binary_data(all_data)

  assert len(sequence) == len(seqName) == len(seqID)

  return sequence, seqName, seqID, label_dict


def generate_dataloader(sequence, seqName, seqID, label_dict, ddp, local_rank, batch_size, num_workers, pin_memory, collate_fn):

    for k, v in label_dict.items():
        Logger.info("label_dict | key: {}: shape: {}".format(k, len(v)))

    dataset = LSDataset(sequence, seqName, seqID, label_dict, local_rank)
    if ddp:
        sampler = DistributedSampler(dataset, num_replicas = torch.distributed.get_world_size(), rank = torch.distributed.get_rank())
        shuffle = None
    else:
        sampler = None
        shuffle = True
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=shuffle,
                            drop_last=False,
                            collate_fn=collate_fn,
                            sampler=sampler)
    return dataloader


def make_loader(local_rank,
                dataset_file,
                split,
                batch_size,
                cfg,
                pin_memory=False,
                num_workers=1,
                random_seed=0):
    
    Logger.info(f'start loading features.')

    sequence, seqName, seqID, label_dict = load_raw_seq(dataset_file,cfg)
    # ipdb.set_trace()
    # name2id = {k:v for k,v in zip(seqName,seqID)}
    
    Logger.info(f"{split} dataloader......")
      
    dataloader = generate_dataloader(sequence, seqName, seqID, label_dict, cfg.mode.ddp, local_rank, batch_size, num_workers, pin_memory, collate_fn)
      
    return {split:dataloader}
    
    




"""
def read_fasta(dataset_file,cfg):
  with open(dataset_file, "rU") as f:
      seq_dict = [(record.id, record.seq._data) for record in SeqIO.parse(f, "fasta")]
  seq_df = pd.DataFrame(data=seq_dict, columns=["Id", "Sequence"])
  seq_df['Sequence'] = seq_df['Sequence'].apply(lambda _: str(_,'utf-8'))

  if cfg.other.debug:
      df_data_pos = seq_df.iloc[:10,:]
      df_data_neg = seq_df.iloc[-10:,:]
      seq_df = pd.concat((df_data_pos,df_data_neg),0)

  return seq_df


def make_seq(df_data):
  sequence = df_data['Sequence'].tolist()
  ids = df_data['Id'].tolist()
  labels = []
  for idx in ids:
    if 'UniRef50' in idx:
      labels.append(0)
    else:
      labels.append(1)
      
  sq_=[]
  length = []
  for idx,s in enumerate(sequence):
    # '''
    # logger.info('>>>{}:{}'.format(idx,s))
    if type(s) is not str:
      s = str(s,'utf-8')
    
    length.append([len(s)])
    sq_.append(' '.join(s))
        
  return sq_,labels

def create_datasetFb(df_data,model_name):

  sentences,labels = make_seq(df_data)
  input_ids = []
  attention_masks = []
  # features = []
  # seqID = torch.tensor([i for i in range(df_data.shape[0])])
  seqID = [i for i in range(df_data.shape[0])]
  seqName = df_data['Id'].to_numpy().ravel()
  tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
  model = AutoModel.from_pretrained(model_name)
  max_length = 200
  # ipdb.set_trace()
  for sent in sentences:
  
      # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
      encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_length,           # Pad & truncate all sentences.
                        truncation = True,
                        padding = 'max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )
      
      attention_masks.append(encoded_dict['attention_mask'])
      input_ids.append(encoded_dict['input_ids'])
  # input_ids = torch.cat(input_ids, dim=0)
  # attention_masks = torch.cat(attention_masks, dim=0)
  labels = {'binary':torch.tensor(labels)}
  
  dataset = LSDataset(input_ids, attention_masks, labels, seqID)
  return dataset,seqName

def create_dataset(df_data,model_name):

  sentences,labels = make_seq(df_data)
  input_ids = []
  attention_masks = []
  
  seqID = torch.tensor([i for i in range(df_data.shape[0])])
  seqName = df_data['Id'].to_numpy().ravel()
  tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
  max_length = 200
  for sent in sentences:
  
      # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
      encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_length,           # Pad & truncate all sentences.
                        truncation = True,
                        padding = 'max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )
    
      # encoded_dict['input_ids'] = torch.where(encoded_dict['input_ids']==24,0,encoded_dict['input_ids'])
      # ipdb.set_trace()
      attention_masks.append(encoded_dict['attention_mask'])
      input_ids.append(encoded_dict['input_ids'])
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  labels = torch.tensor(labels)
  
  dataset = TensorDataset(input_ids, attention_masks, labels, seqID)
  return dataset,seqName
    
# def create_logger_and_dirs(filedir):
def create_logger_and_dirs():

  import logging
  from logging.handlers import TimedRotatingFileHandler
  import sys
  
  # if not os.path.exists(filedir):
  #   os.mkdir(filedir)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  screen_handler = logging.StreamHandler(sys.stdout)
  
  screen_handler.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(module)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
  screen_handler.setFormatter(formatter)
  logger.addHandler(screen_handler)
  
  
  # time_now = int(round(time.time() * 1000))
  # time_now = time.strftime("%Y-%m-%d_%H-%M", time.localtime(time_now / 1000))
  # cls_dir = "{}/{}".format(filedir,time_now)
  # if not os.path.exists(cls_dir): os.makedirs(cls_dir)
  
  log_path = os.path.join(cls_dir,"mylog.log")
  # log_path = 'mylog'
  file_handler = TimedRotatingFileHandler(
      filename=log_path, when="MIDNIGHT", interval=1, backupCount=20
  )
  # filename="mylog" suffix setting: mylog.2020-02-25.log
  file_handler.suffix = "%Y-%m-%d.log"
  # file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
  # log_file = logging.FileHandler(os.path.join(cls_dir,'{}.log'.format(filedir)),'a',encoding = 'utf-8')
  file_handler.setLevel(logging.INFO)
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  
  
  return logger# ,cls_dir
    
def create_parser():
  import argparse
  parser = argparse.ArgumentParser(description='setting hyperparameters')
  parser.add_argument('-output_dir', type=str, default='Knowledge_Distillation',help='output dir name')
  parser.add_argument('-train_epoch', type=int, default=200,help='input epochs default 200')
  parser.add_argument('-eval_step', type=int, default=20,help='epochs to evaluate default 200')
  parser.add_argument('-lr', type=float, default=5e-5,help='learning rate default 5e-5')
  parser.add_argument('-dropout', type=float, default=0.1, help='dropout probability deault 0.1')
  parser.add_argument('-easy_debug', action="store_true", default=False, help=' enable easy debug mode')
  parser.add_argument('-plot', action="store_true", default=False, help=' plot logits for each epoch')
  parser.add_argument("--local_rank", type=int, default=-1,help="local_rank for distributed training on gpus")
  parser.add_argument("--no_cuda", action='store_true',help="Whether not to use CUDA when available")
  parser.add_argument('-LearnGene', action="store_true", default=False, help='initial student model parameters')
  parser.add_argument('-batch_size',type=int, default=32, help='batch size default 32')
  parser.add_argument('-student_model',type=str, default='./pytorch_bert_model', help='student model dir')
  parser.add_argument('-teacher_model',type=str, default='./pytorch_bert_model', help='teacher model dir')
  args = parser.parse_args()
  return args    
"""     