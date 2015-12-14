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



output_folder="/home/ubuntu/Documents/Thesis_work/results/19_oct_results/non_linear/sodp_analysis/5min_trend_features/"


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
                #number of values greater than 0.4
                count_threshold_1=count_threshold_1+1;
            
            if val > threshold_2:
                #number of values greater than 0.3
                count_threshold_2=count_threshold_2+1;  
            
            if val > threshold_3:
                #number of values greater than 0.2
                count_threshold_3=count_threshold_3+1;
            
   
    
    feature_rec.append(count_threshold_3);
    global_vocab,index_of_features=cl.fill_global_vocab("count_threshold_"+str(threshold_3), index_of_features, global_vocab)
    
    feature_rec.append(count_threshold_2);
    global_vocab,index_of_features=cl.fill_global_vocab("count_threshold_"+str(threshold_2), index_of_features, global_vocab)
    
    feature_rec.append(count_threshold_1);
    global_vocab,index_of_features=cl.fill_global_vocab("count_threshold_"+str(threshold_1), index_of_features, global_vocab)
    
    
    
    ##### Calculating 30min features for HRV using toolkit VALUES #################
    total_min=30;
    start_time="00:00:00"
    end_time="00:30:00"
    hrv_feature_list_30min="'SDANN|AVNN|rMSSD|pNN50|TOT PWR|VLF PWR|ULF PWR|LF PWR|HF PWR|LF/HF'"
    list_of_list_of30min_features=pr.get_short_term_hrv(hrv_feature_list_30min,rec_name,annotation,start_time,end_time,output_folder)
    
    for list in list_of_list_of30min_features:
        for feature_tuple in list:
            #print("feature_tuple[1] in list is: " + str(feature_tuple[1]))
            #print("feature_tuple[0] in list is: " + str(feature_tuple[0]))
            feature_rec.append(feature_tuple[1])
            #print("apended feature rec value is:  "+ str(feature_tuple[1]))
            #print("apended feature  name is:  "+ str(feature_tuple[0]))
            global_vocab,index_of_features=cl.fill_global_vocab(feature_tuple[0], index_of_features, global_vocab)
    
    
#     ##### Calculating 5min features for HRV using toolkit VALUES #################
#     total_min=30;
#     list_of_list_of5min_features=pr.get_5min_hrv_features( rec_name, annotation, total_min, output_folder)
#     
#     for list in list_of_list_of5min_features:
#         for feature_tuple in list:
#             feature_rec.append(feature_tuple[1])
#             #print("apended feature rec value is:  "+ str(feature_tuple[1]))
#             #print("apended feature  name is:  "+ str(feature_tuple[0]))
#             global_vocab,index_of_features=cl.fill_global_vocab(feature_tuple[0], index_of_features, global_vocab)
    
    
    
    ###### Calculating non-linear features for RR_sec VALUES #################
    ctm_array=[];
    dist_array=[];
    total_min=30;
    x_val_sodp,y_val_sodp,ctm_array,ctm_feature_name,dist_array,dist_feature_name=nlm.calc_sodp_measures(rec_name,RR_sec, radius_array);
        
    ##add value to feature array and name to global vocab
    for val,name in zip(ctm_array,ctm_feature_name):
        feature_rec.append(val)
        global_vocab,index_of_features=cl.fill_global_vocab(name, index_of_features, global_vocab)
    
    for val,name in zip(dist_array,dist_feature_name):
        feature_rec.append(val)
        global_vocab,index_of_features=cl.fill_global_vocab(name, index_of_features, global_vocab)
    
    #     ##### Calculating 30 min quadrant points ratio features  #################
    start_time=0
    end_time=30;
    feature_list_30min,feature_name=nlm.calc_30min_sodp_measures(rec_name,annotation, start_time,end_time, x_val_sodp,y_val_sodp)
    
    ##add value to feature array and name to global vocab
    for val,name in zip(feature_list_30min,feature_name):
        feature_rec.append(val)
        #print name
        #print val
        global_vocab,index_of_features=cl.fill_global_vocab(name, index_of_features, global_vocab)
    



#     plt.figure()
#     plt.plot(radius_array,ctm_array)
#     plt.title("ctm_array for " +str(rec_name))
#     plt.figure()
#     plt.plot(radius_array,dist_array)
#     plt.title("dist_array for " +str(rec_name))
    
    ##### Calculating std of 5min features in all 6 intervals in 30min sodp features #################
    list_of_listall_features_6_intervals,std_dev_all_features,feature_name_overall=nlm.calc_std_5min_sodp_measures(rec_name,annotation,30, radius_array)
    for val,name in zip(std_dev_all_features,feature_name_overall):
        feature_rec.append(val)
        global_vocab,index_of_features=cl.fill_global_vocab(name, index_of_features, global_vocab)
############################################################################################################################################################


   
#     print("feature list for 5 min RR int is: ")
#     print(feature_list_5min)
#     
#     print ("----------------------------")
#     
#     print("feature_name list is: ")
#     print(feature_name)
    #ctm_list_list.append(ctm_array)
    #dist_list_list.append(dist_array) 
    
    
    
#     even_or_odd=ws.check_rec_even_odd(record)
#     if "o" in even_or_odd:
#         prev_x_val=x_val_sodp
#         prev_y_val=y_val_sodp
        
    #elif "e" in even_or_odd:
        #fig_diff,plot_diff_sodp=graphs.calc_sodp_patient_diff(rec_name,x_val_sodp,prev_x_val,y_val_sodp,prev_y_val)
        #graphs.plotScatter(rec_name,x_val_sodp,y_val_sodp, "x[n+1]-x[n]", " x[n+2]-x[n+1] ", "SODP plot ", 'r',xlim_lo=-1, xlim_hi=1, ylim_lo=-1, ylim_hi=1,axline=1)
        #fig_diff.savefig(output_folder+"diff_sodp_plot_"+record+".png")
        
#     ######### PLOTTING SODP plots array ################
#       
#     if "n" in record:
#           
#         ####plot sodp and save fig ####
#         fig_sodp,plot_sodp=graphs.plotScatter(rec_name,x_val_sodp,y_val_sodp, "x[n+1]-x[n]", " x[n+2]-x[n+1] ", "SODP plot ", 'b',xlim_lo=-1, xlim_hi=1, ylim_lo=-1, ylim_hi=1,axline=1)
#         fig_sodp.savefig(output_folder+"sodp_plot_"+record+".png")
#          
#     elif "p" in record:   
#     ####plot sodp and save fig ####
#         fig_sodp,plot_sodp=graphs.plotScatter(rec_name,x_val_sodp,y_val_sodp, "x[n+1]-x[n]", " x[n+2]-x[n+1] ", "SODP plot ", 'r',xlim_lo=-1, xlim_hi=1, ylim_lo=-1, ylim_hi=1,axline=1)
#         fig_sodp.savefig(output_folder+"sodp_plot_"+record+".png")
#     ######### PLOTTING windowed standard dev array ################
#     
# #     if "n" in record:
# #         
# #         #####plot window std_dev and save fig ####
# #         #fig_std,plot_std=graphs.plot_simple(rec_name,range(len(std_dev_window_arr)), std_dev_window_arr, "window std dev for  window size:  %s" % window_size, "std_dev", "window std dev", "b", 0, 0,-0.1, 0.5)
# #         #fig_std.savefig(output_folder+"std_dev_window_"+record+".png")
# #         
# #         
# #         #write to different file
# #         #rw.write_value(mean_global,output_folder,"normal_mean.txt",write_or_append_n)
# #         #write_or_append_n='a';
# #         
# #     elif "p" in record:
# #             #####plot window std_dev and save fig ####
# #             fig_std_p,plot_std_p=graphs.plot_simple(rec_name,range(len(std_dev_window_arr)), std_dev_window_arr, "window std dev for  window size:  %s" % window_size, "std_dev", "window std dev", "r", 0, 0,-0.1, 0.5)
# #             #fig_std_p.savefig(output_folder+"std_dev_window_"+record+".png")
# #             
# #             
# #             #write to separate file
# #             #rw.write_value(mean_global,output_folder,"patient_mean.txt",write_or_append_p)
# #             #write_or_append_p='a';
# #         
# #     
#     #################### USE filter FUNC TO CLEAN RR_SEC ARRAY  ########################################
#      
#     ##apply square filt 
#     RR_sec_clean_one=dc.square_filt(RR_sec);
#      
#     
#     ##apply quotient filter 
#     RR_sec_clean_two=dc.quotient_filt(RR_sec_clean_one);
# 
#     
#     ##apply quotient filter 2nd time
#     RR_sec_clean=dc.quotient_filt(RR_sec_clean_two);
#     
#     ##comment june 16
#     #print(" np.stdRR_sec_clean is: "+ np.std(RR_sec_clean));
#       
#     
#      
#     ######### detrend  RR_sec_clean array with detrend_data func ###################################
#     RR_sec_detrended=dc.detrend_data(RR_sec_clean);
#     
#       
#     #convert RR_sec_detrended in ms then later calc std
#     RR_sec_detrended_ms=[x * 1000 for x in RR_sec_detrended];
#     
#     
#     ############ Calculate SDRR of clean data #########################
#     
#     #RR_sec_detrended is in ms and std_dev will be in ms
#     sdrr_curr=np.std(RR_sec_detrended_ms);

    #print("SDRR after filtering and detrending in ms for " + rec_name + "is: " + str(sdrr_curr));
    
    
    
    #################### PLOT histograms for each record  ##############################################
#     fig_hist,plot_hist=graphs.plotHistPercent(rec_name, RR_sec,100, "RR (sec)", "Percentage of total points", "RR interval histogram")
#     fig_hist.savefig(output_folder+"hist_"+record+".png")
#     plt.close();

    ################### Append feature for 1 record to a list of lists ##############################
    all_features.append(feature_rec)


#nlm.plot_ctm(rec_name_array,ctm_list_list,radius_array)
#nlm.plot_dist(rec_name_array,dist_list_list,radius_array)
rw.write_value(global_vocab, output_folder, 'global_vocab.txt', 'w')
print(len(all_features))


#write to all_features to file
rw.write_value(all_features,output_folder,"all_features_readable.txt",'w')
rw.write_features_to_file(all_features,output_folder,"all_features_pickle.txt") 
rw.write_features_to_file(global_vocab,output_folder,"global_vocab_pickle.txt")
rw.write_features_to_file(rec_name_array, output_folder, "rec_name_array_pickle.txt")
#plt.show()


