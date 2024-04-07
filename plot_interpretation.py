import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TaiChiACVP.visual.plot import plot_interpret
import ipdb


# Load the data from the Excel file
file_path = './Todraw-CGPT.xlsx'
sheet_names = pd.ExcelFile(file_path).sheet_names

# Load data from each sheet
heatmap_data = pd.read_excel(file_path, sheet_name=sheet_names[0],index_col=0)
annotations_data = pd.read_excel(file_path, sheet_name=sheet_names[1],index_col=0)



plot_interpret(heatmap_data,annotations_data)