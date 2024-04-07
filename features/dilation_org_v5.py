from sklearn.decomposition import PCA
import numpy as np
import torch
from tqdm import tqdm
import torch
import ipdb
from ..utils.std_logger import Logger



class PCXA(object):
    def __init__(self,k,ptype,comp,percent):

        self.pcxa = {
            'layerp': PCA(n_components=k),
            'layern': PCA(n_components=k),
            'normal': { i: PCA(n_components=0.95)
            for i in range(k)},
            'mqPep': { i: PCA(n_components=0.95)
            for i in range(k)},
            'comparison':None
              }
        
        if percent > 0:
          self.pcxa['comparison']= { i: PCA() if percent == 1 else PCA(n_components=percent)
            for i in range(k)}
            
        self.k = k
        self.ptype = ptype
        self.comp = comp
        self.percent = percent
        
    def fit(self, dataset, labels):
    
        Logger.info('fitting pcxa...')
        
        TopKLayers = dataset[:self.k].numpy()# n_layers x n_samples x hidden_dim
        
        PosMask = labels == 1
        NegMask = labels == 0
      
        Logger.info(f'the number of positive samples:{sum(PosMask)}')
        Logger.info(f'the number of negative samples:{sum(NegMask)}')
        
        
        TopKLayers_mean_p = TopKLayers[:,PosMask,:].mean(1)
        TopKLayers_mean_n = TopKLayers[:,NegMask,:].mean(1)
        
        Logger.info(f'the shape of TopKLayers_mean_p:{TopKLayers_mean_p.shape}')
        Logger.info(f'the shape of TopKLayers_mean_n:{TopKLayers_mean_n.shape}')
        
        self.TopKLayers_mean_p = TopKLayers_mean_p
        self.TopKLayers_mean_n = TopKLayers_mean_n
        
        self.pcxa['layerp'].fit(TopKLayers_mean_p.T)
        self.pcxa['layern'].fit(TopKLayers_mean_n.T)
        
        # ipdb.set_trace()
        
        TransLayers_p = np.zeros_like(TopKLayers)
        TransLayers_n = np.zeros_like(TopKLayers)
        
        TransLayers_p2 = np.zeros_like(TopKLayers)
        TransLayers_n2 = np.zeros_like(TopKLayers)
              
        for sample in range(TopKLayers.shape[1]):
        
            if self.ptype == 'taichi' or self.ptype == 'mqPepv2':
              TransLayers_p[:,sample,:] = self.pcxa['layerp'].transform(self.pcxa['layern'].transform(TopKLayers[:,sample,:].T)).T
              TransLayers_n[:,sample,:] = self.pcxa['layern'].transform(self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T)).T
            
              
            if self.ptype == 'pp+nn' :
              TransLayers_p[:,sample,:] = self.pcxa['layerp'].transform(self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T)).T
              TransLayers_n[:,sample,:] = self.pcxa['layern'].transform(self.pcxa['layern'].transform(TopKLayers[:,sample,:].T)).T
            
            if self.ptype == 'np+pn+p+n':
              TransLayers_p[:,sample,:] = self.pcxa['layerp'].transform(self.pcxa['layern'].transform(TopKLayers[:,sample,:].T)).T
              TransLayers_n[:,sample,:] = self.pcxa['layern'].transform(self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T)).T
            
              TransLayers_p2[:,sample,:] = self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T).T
              TransLayers_n2[:,sample,:] = self.pcxa['layern'].transform(TopKLayers[:,sample,:].T).T
              
            if self.ptype == 'nppn+pnnp':
              TransLayers_p[:,sample,:] = self.pcxa['layern'].transform(self.pcxa['layerp'].transform(self.pcxa['layerp'].transform(self.pcxa['layern'].transform(TopKLayers[:,sample,:].T)))).T
              TransLayers_n[:,sample,:] = self.pcxa['layerp'].transform(self.pcxa['layern'].transform(self.pcxa['layern'].transform(self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T)))).T
            
            if self.ptype == 'np+n' :
              TransLayers_p[:,sample,:] = self.pcxa['layerp'].transform(self.pcxa['layern'].transform(TopKLayers[:,sample,:].T)).T
              TransLayers_n[:,sample,:] = self.pcxa['layern'].transform(TopKLayers[:,sample,:].T).T
              
            if self.ptype == 'pn+p' :
              TransLayers_p[:,sample,:] = self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T).T
              TransLayers_n[:,sample,:] = self.pcxa['layern'].transform(self.pcxa['layern'].transform(TopKLayers[:,sample,:].T)).T
            
            if self.ptype == 'np' :
              TransLayers_p[:,sample,:] = self.pcxa['layerp'].transform(self.pcxa['layern'].transform(TopKLayers[:,sample,:].T)).T
              TransLayers_n = None
              
            if self.ptype == 'pn' :
              TransLayers_p = None
              TransLayers_n[:,sample,:] = self.pcxa['layern'].transform(self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T)).T
              
            if self.ptype == 'pcxa' or self.ptype == 'mqPepv1':
              TransLayers_p[:,sample,:] = self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T).T
              TransLayers_n[:,sample,:] = self.pcxa['layern'].transform(TopKLayers[:,sample,:].T).T
        
        if TransLayers_p is None:
          PCXs = TransLayers_n
          
        elif TransLayers_n is None:
          PCXs = TransLayers_p
        elif len(self.ptype.split('+')) < 3:
          PCXs = np.concatenate([TransLayers_n,TransLayers_p],-1)
        else:
          PCXs = np.concatenate([TransLayers_n,TransLayers_p, TransLayers_n2,TransLayers_p2],-1)
        
        if 'mqPep' in self.ptype:
          mqPep = TransLayers_n-TransLayers_p
          PCXs = mqPep # np.concatenate([TransLayers_n,TransLayers_p,mqPep],-1)
          
          
        Logger.info(f'the shape of PCXs:{PCXs.shape}')
        
        for _ in range(self.k):
            self.pcxa['normal'][_].fit(PCXs[_])
        
        if self.percent > 0:
          for _ in range(self.k):
              self.pcxa['comparison'][_].fit(PCXs[_])
            
        return self.pcxa

    
    def transform(self, dataset):

        TopKLayers = dataset[:self.k].numpy()

       
        TransLayers_p = np.zeros_like(TopKLayers)
        TransLayers_n = np.zeros_like(TopKLayers)
        
        TransLayers_p2 = np.zeros_like(TopKLayers)
        TransLayers_n2 = np.zeros_like(TopKLayers)
        
        for sample in range(TopKLayers.shape[1]):
            if self.ptype == 'taichi':
              TransLayers_p[:,sample,:] = self.pcxa['layerp'].transform(self.pcxa['layern'].transform(TopKLayers[:,sample,:].T)).T
              TransLayers_n[:,sample,:] = self.pcxa['layern'].transform(self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T)).T
            
            if self.ptype == 'pp+nn' :
              TransLayers_p[:,sample,:] = self.pcxa['layerp'].transform(self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T)).T
              TransLayers_n[:,sample,:] = self.pcxa['layern'].transform(self.pcxa['layern'].transform(TopKLayers[:,sample,:].T)).T
            
            if self.ptype == 'np+pn+p+n':
              TransLayers_p[:,sample,:] = self.pcxa['layerp'].transform(self.pcxa['layern'].transform(TopKLayers[:,sample,:].T)).T
              TransLayers_n[:,sample,:] = self.pcxa['layern'].transform(self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T)).T
            
              TransLayers_p2[:,sample,:] = self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T).T
              TransLayers_n2[:,sample,:] = self.pcxa['layern'].transform(TopKLayers[:,sample,:].T).T
            
            if self.ptype == 'nppn+pnnp':
              TransLayers_p[:,sample,:] = self.pcxa['layern'].transform(self.pcxa['layerp'].transform(self.pcxa['layerp'].transform(self.pcxa['layern'].transform(TopKLayers[:,sample,:].T)))).T
              TransLayers_n[:,sample,:] = self.pcxa['layerp'].transform(self.pcxa['layern'].transform(self.pcxa['layern'].transform(self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T)))).T
              
            if self.ptype == 'np+n' :
              TransLayers_p[:,sample,:] = self.pcxa['layerp'].transform(self.pcxa['layern'].transform(TopKLayers[:,sample,:].T)).T
              TransLayers_n[:,sample,:] = self.pcxa['layern'].transform(TopKLayers[:,sample,:].T).T
              
            if self.ptype == 'pn+p' :
              TransLayers_p[:,sample,:] = self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T).T
              TransLayers_n[:,sample,:] = self.pcxa['layern'].transform(self.pcxa['layern'].transform(TopKLayers[:,sample,:].T)).T
            
            if self.ptype == 'np' :
              TransLayers_p[:,sample,:] = self.pcxa['layerp'].transform(self.pcxa['layern'].transform(TopKLayers[:,sample,:].T)).T
              TransLayers_n = None
              
            if self.ptype == 'pn' :
              TransLayers_p = None
              TransLayers_n[:,sample,:] = self.pcxa['layern'].transform(self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T)).T
              
            if self.ptype == 'pcxa' or self.ptype == 'mqPepv1':
              TransLayers_p[:,sample,:] = self.pcxa['layerp'].transform(TopKLayers[:,sample,:].T).T
              TransLayers_n[:,sample,:] = self.pcxa['layern'].transform(TopKLayers[:,sample,:].T).T
          
        if TransLayers_p is None:
          PCXs = TransLayers_n
          
        elif TransLayers_n is None:
          PCXs = TransLayers_p
        elif len(self.ptype.split('+')) < 3:
          PCXs = np.concatenate([TransLayers_n,TransLayers_p],-1)
        else:
          PCXs = np.concatenate([TransLayers_n,TransLayers_p, TransLayers_n2,TransLayers_p2],-1)
        
        if 'mqPep' in self.ptype:
          mqPep = TransLayers_n-TransLayers_p
          PCXs = mqPep # np.concatenate([TransLayers_n,TransLayers_p,mqPep],-1)
        
        if self.comp:
        
          if self.percent>0:
            PCXAs = [self.pcxa['comparison'][_].transform(PCXs[_]) for _ in range(self.k)]
          else:
            PCXAs = [PCXs[_] for _ in range(self.k)]
            
        else:
          PCXAs = [self.pcxa['normal'][_].transform(PCXs[_]) for _ in range(self.k)]
        
        return PCXAs