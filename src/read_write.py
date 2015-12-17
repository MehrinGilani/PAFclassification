import matplotlib;
import math;
import matplotlib.pyplot as plt;
import numpy as np;
import wfdb,sys,re;
import scipy;
import scipy.sparse
import scipy.stats
import numpy.linalg as LA;
import pywt;
import os;


from scipy import signal;
from _wfdb import calopen, aduphys;
from wfdb import WFDB_Siginfo
from matplotlib.lines import lineStyles
from _wfdb import strtim
from matplotlib.pyplot import show
from numpy import dtype, argmin 
from scipy.stats.mstats_basic import kurtosistest
from matplotlib.ticker import FuncFormatter
import pickle 
import pandas as pd


def write_value(value,path_to_file,file_name,append_or_write):
    #this function appends or writes values to files line by line
    if append_or_write == 'a':
        f=open(path_to_file+file_name,'a');
        f.write(str(value)+"\n")
        f.close
    if append_or_write == 'w':
        f=open(path_to_file+file_name,'w');
        f.write(str(value)+"\n")
        f.close
    return None


def write_features_to_file(list_of_lists,path_to_file,file_name):
    #this file uses pickle to dump list of lists to text file
    with open(path_to_file+file_name,"w") as internal_filename:
        pickle.dump(list_of_lists,internal_filename)
    return None

def read_features_frm_file(output_folder,file_name):
    #open output folder and output file, read the list in the file and append in array
    with open(output_folder+file_name,"r") as new_filename:
        all_feature_array=pickle.load(new_filename)
    return all_feature_array

def combine_n_write_df_to_csv(feature_matrix,feature_names,col_to_add,added_col_header,output_folder,file_name):
    #this function stores the features with their headers in csv file
    feature_df=pd.DataFrame(feature_matrix)
    #feature_df.loc[:,added_col_header]=pd.Series(col_to_add,index=feature_df.index)
    feature_df.loc[:,added_col_header[0]]=pd.Series(col_to_add)
    file_path=output_folder+file_name
    #print(feature_names)
    #print(added_col_header)
    all_headers=list(feature_names)+list(added_col_header)
    #print all_headers
    feature_df.to_csv(file_path,header=all_headers,index=False)
    return None
 
def write_df_to_csv(feature_matrix,feature_names,output_folder,file_name):
    #this function stores the features with their headers in csv file
    feature_df=pd.DataFrame(feature_matrix)
    file_path=output_folder+file_name
    feature_df.to_csv(file_path,header=feature_names,index=False)
    return None   



    