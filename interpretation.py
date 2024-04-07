import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig
from TaiChiACVP.utils.std_logger import Logger
from TaiChiACVP.features.loadpcxa_org_v5 import load_and_dilation
from TaiChiACVP.interpretor.loadprop import load 
from TaiChiACVP.visual.plot import *

@hydra.main(config_path="configs", config_name="train_org.yaml")
def main(cfg: DictConfig):
    # ipdb.set_trace()
    orig_cwd = hydra.utils.get_original_cwd()
    cfg.orig_cwd = orig_cwd
    local_rank = 0
    global_rank = 0
    
    if cfg.features.peptide.HC:
        HC_training_sets, HC_test_sets = {}, {}
        
        for nlab in cfg.data.files_list: 

            HC_train = pd.read_csv(f"/public1/home/scb6744/PCBERTA/data/{nlab}-train.csv",index_col=0)
            HC_test = pd.read_csv(f"/public1/home/scb6744/PCBERTA/data/{nlab}-test.csv",index_col=0)
            
            features_inds = HC_train.columns[2:-2]
            # ipdb.set_trace()
            # HC_training_sets[features][nlab] = HC_train[features_inds].values
            HC_test_sets[nlab] = HC_test[features_inds].values
    
    if cfg.features.peptide.LLM:
        training_sets, test_sets,training_labels, test_labels = load_and_dilation(cfg)
    
    Prop_test_sets = load(cfg)

    # ipdb.set_trace()
    for k,features in zip(cfg.features.pretrained.dilation.topK, cfg.features.pretrained.dilation.features_list):
        # ipdb.set_trace()
        all_perfs = {}
        for nlab in cfg.data.files_list:

            nlab_dir = nlab
            os.makedirs(nlab_dir,exist_ok=True)
            
            Prop_df = Prop_test_sets[nlab].iloc[:,2:-2]
            new_columns = [f'Prop-{n}' for n in Prop_df.columns]
            Prop_df.columns = new_columns
            for model in  cfg.features.peptide.keys():

                if model == 'LLM' and cfg.features.peptide[model]:
                    for layer in range(len(training_sets[features][nlab])):
                
               
                

                        Logger.info("analysing pcxa...")
                        name = f'pc{features}{k}a-{layer}'
                        # X_train_list.append(training_sets[features][nlab][layer])
                        LLM = test_sets[features][nlab][layer]
                        LLM_df = pd.DataFrame(LLM,columns=[f"{name}-{_}d" for _ in range(LLM.shape[1])],index = Prop_df.index)
                        # ipdb.set_trace()
                        pearson_corr = pd.concat([Prop_df, LLM_df],1).corr(method='pearson').iloc[Prop_df.shape[1]:,:Prop_df.shape[1]]
                        heatmap_plot(pearson_corr,f"{nlab}-pearson of properties with {name}")
                        pearson_corr.to_csv(f"{nlab_dir}/{nlab}-pearson of properties with {name}.csv")
                        # ipdb.set_trace()
                if model == 'HC' and cfg.features.peptide[model]:

                    Logger.info("analysing HC...")
                    
                    name = 'HC'
                    HC_df = pd.DataFrame(HC_test_sets[nlab],index = Prop_df.index, columns = features_inds)
                    # ipdb.set_trace()
                    # X_train_list.append(HC_training_sets[features][nlab])
                    pearson_corr = pd.concat([Prop_df, HC_df],1).corr(method='pearson').iloc[Prop_df.shape[1]:,:Prop_df.shape[1]]
                    heatmap_plot(pearson_corr,f"{nlab}-pearson of properties with {name}",square=False)

                    pearson_corr.to_csv(f"{nlab_dir}/{nlab}-pearson of properties with {name}.csv")
            


                

if __name__=='__main__':
    main()
    print('Done.')


    