import pandas as pd
from scipy.stats import ranksums

def feature_selection_ranksum(df_1,df_2,features_ind):
    # Column index should be the same at df_1 and df_2
    # assert False not in (df_1.columns == df_2.columns)
    # Leave only the features of peptides here
    ind_selected,ind_bert_selected = [],[]
    
    for fi in features_ind:
        pval = ranksums(df_1[fi], df_2[fi]).pvalue
        # pval_features.append(pval)
        if pval < .05:
            ind_selected.append(fi)
    
    return ind_selected