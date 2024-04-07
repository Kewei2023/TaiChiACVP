import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import ipdb
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.tree import plot_tree
import os

def heatmap_plot(feature,name,figsize=(20,20),square=True):
  
  fig = plt.figure(figsize=figsize,dpi=300)
  # plt.rc('font', family='Times New Roman')
  plt.rcParams["font.weight"] = "bold"
  plt.rcParams["axes.labelweight"] = "bold"
  h = sns.heatmap(feature,cbar=True,annot=True,fmt=".2f",square=square,annot_kws={'fontsize':'small'})
#   cb = h.figure.colorbar(h.collections[0]) 
#   cb.ax.tick_params(labelsize=28)  

  plt.title(name)
  plt.savefig(f'{name}.png')
  plt.clf()

def plot_interpret(values, annotations):

  plt.figure(figsize=(12, 9))
  # plt.rcParams['font.family'] = 'Times New Roman'
  plt.rcParams['font.size'] = 12
  # Draw the heatmap with the mask and correct aspect ratio
  sns.heatmap(values, annot=annotations, fmt='', cmap='viridis', linewidths=.5, cbar_kws={"shrink": .5})
  
  # Add labels and title for clarity
  plt.title('The physicochemical meaning of each TaiChiNet features', pad=20)
  plt.xlabel('Dataset')
  plt.ylabel('TaiChiNet features')
  
  # Show the heatmap
  plt.tight_layout() # Adjust layout to fit everything
  
  plt.savefig('show interpret.png')


def plot_trees(feature_names,interested_feature_name,model,savedir,name): # e.g. interested_features = [0,2]
  
  if interested_feature_name is None:
    interested_features = feature_names
  else:
    interested_features = [index for index, element in enumerate(feature_names) if interested_feature_name in element]
  
  name_str = '-'.join(name.split('/')[-2:])
  # ipdb.set_trace()
  # create PDF file
  with PdfPages(os.path.join(savedir,f'{name_str}-forest_trees.pdf')) as pdf:
      # find and visualize the tree
      for idx, estimator in enumerate(model.estimators_):
          # get all index in the tree
          features_used = estimator.tree_.feature
          # check the tree use the interested features
          if any(feature in interested_features for feature in features_used if feature != -2):
              # visualize the tree
              plt.figure(figsize=(20, 10))
              plot_tree(estimator,
                        feature_names=feature_names,
                        filled=True)
              plt.title(f'{name_str}-Tree {idx}')
              pdf.savefig()  
              plt.close()  





def calculate_angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    # Convert the angle from radians to degrees
    angle_degree = np.degrees(angle)
    return angle_degree

  
def plot_pca(nlab,pos_mean_sample, neg_mean_sample, pcxa):
  
  result={}
  pos_inter_pca, neg_inter_pca = pcxa['layerp'], pcxa['layern']
  
  origin_pos = pos_mean_sample.copy()
  origin_neg = neg_mean_sample.copy()
  
  
  # pcberta
  pcberta_pp = pos_inter_pca.transform(origin_pos.T).T
  pcberta_pn = pos_inter_pca.transform(origin_neg.T).T
  
  # ipdb.set_trace()
  
  pcberta_np = neg_inter_pca.transform(origin_pos.T).T
  pcberta_nn = neg_inter_pca.transform(origin_neg.T).T
  
  # ipdb.set_trace()
  
  pcberta_p = np.concatenate([pcberta_pp,pcberta_np],1)
  pcberta_n = np.concatenate([pcberta_pn,pcberta_nn],1)
  
  origin_all= np.concatenate([origin_pos,origin_neg],1)
  pcberta_all= np.concatenate([pcberta_p,pcberta_n],1)
 
  
  
  # taichinet
  taichi_pp = neg_inter_pca.transform(pos_inter_pca.transform(origin_pos.T)).T
  taichi_pn = neg_inter_pca.transform(pos_inter_pca.transform(origin_neg.T)).T
  
  taichi_np = pos_inter_pca.transform(neg_inter_pca.transform(origin_pos.T)).T
  taichi_nn = pos_inter_pca.transform(neg_inter_pca.transform(origin_neg.T)).T
  
  taichi_p = np.concatenate([taichi_pp,taichi_np],1)
  taichi_n = np.concatenate([taichi_pn,taichi_nn],1)
  
  
  pos_matrix = (pos_inter_pca.components_.T)
  neg_matrix = (neg_inter_pca.components_.T)
  
  taichi_pos_matrix = pos_matrix @ neg_matrix
  taichi_neg_matrix = neg_matrix @ pos_matrix
  
  ipdb.set_trace()
  # start drawing
  fig, axes = plt.subplots(1, 3, figsize=(18, 6))
  
  
  # define color
  color_scheme = {
    'dark_blue': '#003366',
    'light_blue': '#6699CC',
    'dark_red': '#990000',
    'light_red': '#FF6666',
    'dark_green': '#006633',
    'light_green': '#66CC99',
    'dark_yellow': '#CC9900',
    'light_yellow': '#FFFF99'
  }
  
  
  axes[0].scatter(origin_pos[0], origin_pos[1],color='white',alpga=0.5,marker='o',label='original data positive')
  axes[0].scatter(origin_neg[0], origin_neg[1],color='black',alpga=0.5,marker='o',label='original data negative')
  axes[0].set_title("Original data")
  axes[0].set_facecolor('lightgray')  
  
  
  
  axes[1].scatter(origin_pos[0], origin_pos[1],color='white',alpga=0.5,marker='o',label='PCBERTA positive')
  axes[1].scatter(origin_neg[0], origin_neg[1],color='black',alpga=0.5,marker='o',label='PCBERTA negative')
  axes[1].set_title("Original data")
  axes[1].set_facecolor('lightgray')  
  arrow_patch, arrow_legend = [], []
  for idx, (v, color) in enumerate(zip(pos_matrix.T, [color_scheme['dark_blue'],color_scheme['light_blue']])):
    arrow = mpatches.FancyArrowPatch(pos_inter_pca.mean_, pos_inter_pca.mean_ + v,
                                       color=color,
                                       mutation_scale=20)
    arrow_patch.append(arrow)
    arraw_legend.append(f'$TaiChiNet^+$ dimension {idx}')                                  
    axes[1].add_patch(arrow)
  
  
  
  
  
