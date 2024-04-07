from .dilation_org_v5 import PCXA
import pandas as pd
import numpy as np
import torch
import os
import ipdb 
from ..utils.std_logger import Logger
from TaiChiACVP.visual.plot import plot_pca
def load_and_dilation(cfg):
      # features_list = ['esm2_t36','esm2_t48','bert-base','bert-large']
      # cls_categories = ["Anti-Virus", "non-AVP", "non-AMP", "All-Neg", "All-AMP"]
      
      training_sets, test_sets,training_labels, test_labels = {}, {}, {}, {}
      
      # if cfg.features.pretrained.dilation.type == 'pcxa':
      PCXAs_training_datasets,PCXAs_test_datasets = {}, {}
      # ipdb.set_trace()
      for k,features in zip(cfg.features.pretrained.dilation.topK, cfg.features.pretrained.dilation.features_list):

          training_sets[features] = {
              
                  nlab: torch.load(os.path.join(cfg.features.pretrained.save.path,'train_data',f'{nlab}-{features}-train-12layers.pt')) 
                  for nlab in cfg.data.files_list
              
              }

          
          training_labels[features] = {

              nlab: pd.read_csv(os.path.join('/public1/home/scb6744/TaiChiACVP/data',f'{nlab}-train.csv'),index_col=0)['label'].values
              for nlab in cfg.data.files_list
          }


          test_sets[features] = {
              
              nlab: torch.load(os.path.join(cfg.features.pretrained.save.path,'test_data',f'{nlab}-{features}-test-12layers.pt'))
              for nlab in cfg.data.files_list
              
          }
          
          test_labels[features] = {

              nlab: pd.read_csv(os.path.join('/public1/home/scb6744/TaiChiACVP/data',f'{nlab}-test.csv'),index_col=0)['label'].values
              for nlab in cfg.data.files_list
          }

          
          # ipdb.set_trace()
            # k need to be defined by pcxa_layer.py
            
            
          PCXAs_training_datasets[features],PCXAs_test_datasets[features] ={},{}
          for nlab in cfg.data.files_list:
            
            Logger.info(f'Loading {nlab}')
            
            
            pcxa = PCXA(k,cfg.features.pretrained.dilation.type,cfg.features.pretrained.dilation.comparison, cfg.features.pretrained.dilation.percentage)

            # fit layer pcas 
            pcxa.fit(training_sets[features][nlab],training_labels[features][nlab])

            PCXAs_training_datasets[features][nlab] = pcxa.transform(training_sets[features][nlab])
            PCXAs_test_datasets[features][nlab] = pcxa.transform(test_sets[features][nlab])
            
            Logger.info(f"{nlab}:the shape of pcxa for each layer:")
            for _ in range(len(PCXAs_training_datasets[features][nlab])):
              Logger.info(f"layer {_}:{PCXAs_training_datasets[features][nlab][_].shape}")
            
            # plot_pca(nlab,pcxa.TopKLayers_mean_p, pcxa.TopKLayers_mean_n, pcxa.pcxa)#['layerp'], pcxa.pcxa['layern'],pcxa.pcxa['normal'])
             
              
      # ipdb.set_trace()
      return PCXAs_training_datasets,PCXAs_test_datasets, training_labels,test_labels

