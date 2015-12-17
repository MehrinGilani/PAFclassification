#this is a test file

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
from sklearn import cross_validation
from sklearn import svm


import wfdb_setup as ws;
import process_rr as pr;
import data_cleaning as dc;
import graphs 
import read_write as rw;
import non_linear_measures as nlm;
import classification_functions as cl



output_folder="/home/ubuntu/Documents/Thesis_work/results/19_oct_results/non_linear/sodp_analysis/afib_normal_data_generation/filtered_patient_afpdb_plot/quo_filt_patient_afpdb/"


db_name="afpdb";
initial_rec_array=[];
rec_name_array=[];
annotator_array=[];
wo_continuation_recs=[]

recs_to_remove=['n24','n27','n28'];

annotator_array=ws.dload_annotator_names(db_name);
for ann_name in annotator_array:
    print ann_name
    if ann_name == "qrs":
        annotation=ann_name;
    if ann_name == "atr":
        annotation=ann_name;
print("annotators for this database are: " + str(annotator_array) + " we are choosing " + str(annotation))
initial_rec_array=ws.dload_rec_names(db_name);


wo_continuation_recs=ws.rmv_continuation_rec(initial_rec_array)
rec_name_array=ws.rmv_test_rec(wo_continuation_recs)
#rec_name_array=ws.rmv_even_rec(wo_continuation_recs)


print str(rec_name_array)

#############################################################################
#### list of features (per record) that we want to extract ######
# global_mean
# global_std_dev
# nn50
# pnn50
# sdsd
# window_1_std_dev
# .
# .
# .
# window_100_std_dev
# std_dev_all_window
# diff_rr_std_dev (might be similar to rmSSD)
### start with time domain for now
# freq domain for 5 mins of data
# total_pwr_1
# .
# .
# .
# total_pwr_6
# VLF_pwr_1
# .
# .
# .
# VLF_pwr_6
# HF_pwr_1
# .
# .
# .
# HF_pwr_6
# LF/HF ratio_1
# .
# .
# .
# LF/HF ratio_6
# ctm_1
# .
# .
# .
# ctm_20
# dist_1
# .
# .
# .
# dist_20

#other features that can be added : 
# poincare_sd1
# poincare_sd2
# poincare_ratio
# ApEn
# SampEn
#### end of list of features (per record) that we want to extract ######
#############################################################################


#radius_array=[0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38, 0.40];
radius_array=[x / 1000.0 for x in range(0, 1000, 20)]
all_features=[]
ctm_list_list=[]
dist_list_list=[]

prev_x_val=[]
prev_y_val=[]

write_or_append_n='w';
write_or_append_p='w';

#dictionary to store indices of features
global_vocab={}; 
index_of_features=0;

for record in rec_name_array:
    mean_global_arr=[]
    feature_rec=[]
    #variables and arrays 
    SDRR=[];    
    SDRR_ms=[];    
    ann_graph=[];
    RR_sec_unclean=[];
    RR_sec=[];    
    delta_RR_sec=[];
    x_val_sodp=[]
    y_val_sodp=[]
    rec_name=db_name+"/"+record;
    #rec_name= "afpdb/p02c"
    
    ## time for which you want to read record ##
    start_time=0;
    end_time=30; #time in mins
    
    ##### Extract RR intervals here #######
    
    RR_sec_unclean=pr.get_RR_interval(rec_name,annotation,start_time,end_time)
    
    ####DELETE THE FIRST RR_sec_unclean value#####
    del RR_sec_unclean[0];
    
    
    
    ##### APPLY FILTERS TO CLEAN DATA #######
    
    #RR_sec=dc.detrend_data(dc.quotient_filt(dc.square_filt(RR_sec_unclean)))
    #RR_sec=dc.detrend_data((RR_sec_unclean)
    
    RR_sec=dc.quotient_filt(dc.square_filt(RR_sec_unclean))
    #RR_sec=dc.square_filt(RR_sec_unclean)
    #RR_sec=RR_sec_unclean
    ##### Extract delta RR intervals here #######
    delta_RR_sec = pr.get_delta_rr(RR_sec);
    
    ###### Calculating GOLBAL statistical features for RR_sec VALUES #################

    ###calculating AVG/mean of RR intervals ###
    mean_global=np.mean(RR_sec)*1000;
    mean_global_arr.append(mean_global) #do we need this?
    feature_rec.append(mean_global)
    global_vocab,index_of_features=cl.fill_global_vocab("mean_global", index_of_features, global_vocab)
    
    
#     if "mean_global" not in global_vocab.keys():
#         global_vocab["mean_global"]=index_of_features;
#         index_of_features=index_of_features+1;
    #print("mean_global is: " + str(mean_global))
    
    
    #sdrr
    sdrr_raw=np.std(RR_sec);
    #print (" sdrr of RR_sec (raw) in sec is: " +str(sdrr_raw));
    
    sdrr_raw_ms=sdrr_raw*1000;
    std_dev_global=sdrr_raw_ms;
    feature_rec.append(std_dev_global)
    global_vocab,index_of_features=cl.fill_global_vocab("std_dev_global", index_of_features, global_vocab)
    
#     print (" sdrr of RR_sec (raw) in ms is: " +str(sdrr_raw_ms));
        
    #### calculating skewness and kurtosis #####
    #skewness_global=scipy.stats.skew(RR_sec)
    #print("skewness: "+ str(skewness_global))
    
    #kurtosis_global=scipy.stats.kurtosis(RR_sec, fisher='false') 
    #print("kurtosis_globl is : " +str(kurtosis_global))   
    #print("answer for kurtosis test is: "+str(scipy.stats.kurtosistest(RR_sec))) 
    
    
    ###### Calculate nn50, pnn50 and rmSSD or diff_rr_std_dev ######
    std_dev_diff=np.std(delta_RR_sec);
    feature_rec.append(std_dev_diff);
    global_vocab,index_of_features=cl.fill_global_vocab("std_dev_diff", index_of_features, global_vocab)
    
    ###### Calculating window based statistical features for RAW SDRR VALUES #################
    window_size=50;
    std_dev_window_arr=[]
    std_dev_window_arr=pr.get_window_std_dev(RR_sec, window_size)
    window_number=1;
#     for val in std_dev_window_arr:
#         feature_rec.append(val)
#         global_vocab,index_of_features=cl.fill_global_vocab("std_window_"+str(window_number), index_of_features, global_vocab)
#         window_number=window_number+1;

###plot window std_dev and save fig ####
    fig_std_p,plot_std_p=graphs.plot_simple(rec_name,range(len(std_dev_window_arr)), std_dev_window_arr, "quo_filt window std dev for  window size:  %s" % window_size, "std_dev", "window std dev", "r", 0, 0,-0.1, 0.5)
    fig_std_p.savefig(output_folder+"std_dev_window_"+record+".png")