from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import torch
import os
import ipdb

def load_llm(cfg):
      # features_list = ['esm2_t36','esm2_t48','bert-base','bert-large']
      # cls_categories = ["Anti-Virus", "non-AVP", "non-AMP", "All-Neg", "All-AMP"]
      
  training_sets, test_sets,training_labels, test_labels = {}, {}, {}, {}
    
    
  for k,features in zip(cfg.features.pretrained.dilation.topK, cfg.features.pretrained.dilation.features_list):
      
    training_sets[features],training_labels[features] = {}, {}
    test_sets[features],test_labels[features] = {}, {}
    
    
    training_sets[features] = {
        
            nlab: torch.load(os.path.join(cfg.features.pretrained.save.path,'train_data',f'{nlab}-{features}-train-12layers.pt')) 
            for nlab in cfg.data.files_list
        
        }

    
    training_labels[features] = {

        nlab: pd.read_csv(os.path.join('/public1/home/scb6744/PCBERTA/data',f'{nlab}-train.csv'),index_col=0)['label'].values
        for nlab in cfg.data.files_list
    }


    test_sets[features] = {
        
        nlab: torch.load(os.path.join(cfg.features.pretrained.save.path,'test_data',f'{nlab}-{features}-test-12layers.pt'))
        for nlab in cfg.data.files_list
        
    }
    
    test_labels[features] = {

        nlab: pd.read_csv(os.path.join('/public1/home/scb6744/PCBERTA/data',f'{nlab}-test.csv'),index_col=0)['label'].values
        for nlab in cfg.data.files_list
    }

      
  return training_sets,test_sets, training_labels,test_labels

