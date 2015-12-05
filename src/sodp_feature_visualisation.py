## this file reads array of all features and does classification_functions


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
from numpy import dtype, argmin, shape
from scipy.stats.mstats_basic import kurtosistest
from matplotlib.ticker import FuncFormatter
from sklearn import preprocessing;
from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets
from sklearn import cross_validation
from sklearn import metrics


import wfdb_setup as ws;
import process_rr as pr;
import data_cleaning as dc;
import graphs 
import read_write as rw;
import non_linear_measures as nlm;
import classification_functions as cl



output_folder="/home/ubuntu/Documents/Thesis_work/results/19_oct_results/non_linear/sodp_analysis/sodp_patient_difference/"

rec_list=['p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'p30', 'p31', 'p32', 'p33', 'p34', 'p35', 'p36', 'p37', 'p38', 'p39', 'p40', 'p41', 'p42', 'p43', 'p44', 'p45', 'p46', 'p47', 'p48', 'p49', 'p50']
#rec_list=['p01', 'p02', 'p03', 'p04', 'p05', 'p06']
i=0
while  i < len(rec_list):
    
    file1=rec_list[i]
    #print("file 1 is " +str(file1))
    file2=rec_list[i+1]
    #print("file 2 is " +str(file2))
    i=i+2
    #print i
    
    nlm.calc_sodp_patient_diff(output_folder,("sodp_plot_"+file2+".png"),("sodp_plot_"+file1+".png"),(file2+"-"+file1+".png"))

exit()


