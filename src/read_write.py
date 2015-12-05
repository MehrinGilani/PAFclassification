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

# def write_feature(rec_name,feature_value,path_to_file,file_name):
#     #this function adds the feature value to the comma separated file
#     f=open(path_to_file+file_name,'wb+');
#     if line in f 
#     f.write(str(feature_value)),","