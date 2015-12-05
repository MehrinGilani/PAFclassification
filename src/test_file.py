##this file acts as the main file and imports functions from other files 
# this file extracts features from raw_rr_sec

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


#for the given db_name
##download record names
##download annotator names

#for every record
##1. Open record 
##2. Extract RR interval
##3. Plot Histogram with constant bins
    ###3.1 y axis of Histogram shows frequency in percentage
##3. Extract Delta RR 


#use main from other file
#from main import svm_classifier function

#also import nonlinear measures from other file
#from



output_folder="/home/ubuntu/Documents/Thesis_work/testing/hrv_5min/"


db_name="afpdb";
initial_rec_array=[];
rec_name_array=[];
annotator_array=[];
wo_continuation_recs=[]

recs_to_remove=['n24','n27','n28'];

annotator_array=ws.dload_annotator_names(db_name);
annotation=annotator_array[0];
print("annotators for this database are: " + str(annotator_array) + " we are choosing " + str(annotator_array[0]))

initial_rec_array=ws.dload_rec_names(db_name);


wo_continuation_recs=ws.rmv_continuation_rec(initial_rec_array)
rec_name_array=ws.rmv_test_rec(wo_continuation_recs)



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
    
    
    #RR_sec=dc.quotient_filt(dc.square_filt(RR_sec_unclean))
    RR_sec=RR_sec_unclean
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
    
    ### calculating variation in the std_dev_window_arr and save as feature
    var_std_dev_window_arr=np.std(std_dev_window_arr);
   # print("type of np.std elemnt is : " + str(type(var_std_dev_window_arr()))
    
    #print("std of std array is " +str(var_std_dev_window_arr))
    #var_std_dev_window_arr=np.nan_to_num(var_std_dev_window_arr)
    #print "standard deviation of var_std_dev_window_arr is" +str(var_std_dev_window_arr)
    feature_rec.append(np.float64(var_std_dev_window_arr));
    global_vocab,index_of_features=cl.fill_global_vocab("var_std_dev_window_arr", index_of_features, global_vocab)

    ### calculating number of times std_dev_array pass a value
    #if value is greater than 0.4 add 1 to count, if not then check the next statment
    threshold_3=0.2;
    threshold_2=0.3;
    threshold_1=0.4;
    count_threshold_1=0;
    count_threshold_2=0;
    count_threshold_3=0;
    
    for val in std_dev_window_arr:
            if val > threshold_1:
                #number of values greater than 0.2
                count_threshold_1=count_threshold_1+1;
            
            elif val > threshold_2:
                #number of values greater than 0.3
                count_threshold_2=count_threshold_2+1;  
            
            elif val > threshold_3:
                #number of values greater than 0.4
                count_threshold_3=count_threshold_3+1;
            
   
    
    feature_rec.append(count_threshold_3);
    global_vocab,index_of_features=cl.fill_global_vocab("count_threshold_"+str(threshold_3), index_of_features, global_vocab)
    
    feature_rec.append(count_threshold_2);
    global_vocab,index_of_features=cl.fill_global_vocab("count_threshold_"+str(threshold_2), index_of_features, global_vocab)
    
    feature_rec.append(count_threshold_1);
    global_vocab,index_of_features=cl.fill_global_vocab("count_threshold_"+str(threshold_1), index_of_features, global_vocab)
    
    
    
    ###### Calculating SDANN and LF/HF ratio features for HRV using toolkit VALUES #################
    start_time="00:00:00"
    end_time="00:30:00"
    
#     SDANN=pr.get_short_term_hrv("SDANN",rec_name,annotation,start_time,end_time,output_folder)
#     print ("SDANN is: " + str(SDANN));
#     feature_rec.append(SDANN);
#     global_vocab,index_of_features=cl.fill_global_vocab("SDANN", index_of_features, global_vocab)
    
    
    ##### Calculating 30min features for HRV using toolkit VALUES #################
    total_min=30;
    feature_list="'SDNN|AVNN|rMSSD|pNN50|TOT PWR|VLF PWR|LF PWR|HF PWR|LF/HF'"
    list_of_list_of30min_features=pr.get_short_term_hrv(feature_list,rec_name,annotation,start_time,end_time,output_folder)
    
    for list in list_of_list_of30min_features:
        for feature_tuple in list:
            #print("feature_tuple[1] in list is: " + str(feature_tuple[1]))
            #print("feature_tuple[0] in list is: " + str(feature_tuple[0]))
            feature_rec.append(feature_tuple[1])
            print("apended feature rec value is:  "+ str(feature_tuple[1]))
            print("apended feature  name is:  "+ str(feature_tuple[0]))
            global_vocab,index_of_features=cl.fill_global_vocab(feature_tuple[0], index_of_features, global_vocab)
    
    
    
    
    