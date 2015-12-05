##ttries to see if we can predict afib for 1 patient
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


output_folder="/home/ubuntu/Documents/Thesis_work/results/19_oct_results/non_linear/sodp_analysis/afib_prediction_sodp/5min_sodp_even_rec/"




db_name="afpdb";
initial_rec_array=[];
rec_name_array=[];
annotator_array=[];
rec_name_arraywo_continuation_recs=[]



annotator_array=ws.dload_annotator_names(db_name);
annotation=annotator_array[0];
print("annotators for this database are: " + str(annotator_array) + " we are choosing " + str(annotator_array[0]))

initial_rec_array=ws.dload_rec_names(db_name);
print initial_rec_array
#rec_name_array=initial_rec_array

#with_continuation_recs=ws.keep_continuation_rec(initial_rec_array)
#rec_name_array=with_continuation_recs
#print(with_continuation_recs)


# wo_continuation_recs=ws.rmv_continuation_rec(initial_rec_array)
# rec_name_array=ws.rmv_test_rec(wo_continuation_recs)
# # rec_name_array=ws.rmv_even_rec(initial_rec_array)
# rec_name_array=ws.rmv_odd_rec(rec_name_array)
# #rec_name_array=with_continuation_recs
# print(rec_name_array)



radius_array=[0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38, 0.40];
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
    
   
    
    ################### SODP plots #########################

    ## time for which you want to read record ##
    start_time_arr=[0,5,10,15,20,25]
    end_time_arr=[5,10,15,20,25,30]
    
    #for continutaoin record we just have 5 mins
    #start_time_arr=[0]
    #end_time_arr=[5]
    for start_time, end_time in zip(start_time_arr,end_time_arr):
    
        print ("start time is: " +str(start_time));
        print ("end time is: " +str(end_time));
        
        
        ##### Extract RR intervals here #######
        
        RR_sec_unclean=pr.get_RR_interval(rec_name,annotation,start_time,end_time)
        
        ####DELETE THE FIRST RR_sec_unclean value#####
        del RR_sec_unclean[0];
        
        RR_sec=RR_sec_unclean
        
    # #     #### APPLY FILTERS TO CLEAN DATA #######
    # #     
    # #     RR_sec=dc.detrend_data(dc.quotient_filt(dc.square_filt(RR_sec_unclean)))
    # #     RR_sec=dc.quotient_filt(dc.square_filt(RR_sec_unclean))
    # #     RR_sec=RR_sec_unclean
    # #     #### Extract delta RR intervals here #######
    # #     delta_RR_sec = pr.get_delta_rr(RR_sec);
        
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
        print (" sdrr of RR_sec (raw) in ms is: " +str(sdrr_raw_ms));
        
        ##### Calculating non-linear features for RR_sec VALUES #################
        ctm_array=[];
        dist_array=[];
       
        x_val_sodp,y_val_sodp,ctm_array,dist_array=nlm.calc_sodp_measures(rec_name,RR_sec, radius_array);
        ctm_list_list.append(ctm_array)
        dist_list_list.append(dist_array) 
#         ######### PLOTTING SODP for afib plots array ################
#           
#         if "n" in record:
#               
#             ####plot sodp and save fig ####
#             fig_sodp,plot_sodp=graphs.plotScatter(rec_name,x_val_sodp,y_val_sodp, "x[n+1]-x[n]", " x[n+2]-x[n+1] ", "SODP plot for "+str(start_time)+"to"+str(end_time), 'b',xlim_lo=-1, xlim_hi=1, ylim_lo=-1, ylim_hi=1,axline=1)
#             fig_sodp.savefig(output_folder+"sodp_plot_"+record+"_"+str(start_time)+"to"+str(end_time)+".png")
#              
#         elif "p" in record:   
#         ####plot sodp and save fig ####
#             fig_sodp,plot_sodp=graphs.plotScatter(rec_name,x_val_sodp,y_val_sodp, "x[n+1]-x[n]", " x[n+2]-x[n+1] ", "SODP plot for "+str(start_time)+"to"+str(end_time), 'r',xlim_lo=-1, xlim_hi=1, ylim_lo=-1, ylim_hi=1,axline=1)
#             fig_sodp.savefig(output_folder+"sodp_plot_"+record+"_"+str(start_time)+"to"+str(end_time)+".png")
#         
        ######### PLOTTING SODP for afib plots array ################
           
       
               
        ####plot sodp and save fig ####
        fig_sodp,plot_sodp=graphs.plotScatter(rec_name,x_val_sodp,y_val_sodp, "x[n+1]-x[n]", " x[n+2]-x[n+1] ", "SODP plot for "+str(start_time)+"to"+str(end_time), 'b',xlim_lo=-1, xlim_hi=1, ylim_lo=-1, ylim_hi=1,axline=1)
        fig_sodp.savefig(output_folder+"sodp_plot_"+record+"_"+str(start_time)+"to"+str(end_time)+".png")
          
        
         

