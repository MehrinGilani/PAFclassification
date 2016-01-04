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
from scipy.stats import describe
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

import process_ecg as pecg


# plt.figure()
# x=range(0,10)
# y=x
# plt.scatter(x,y)
# plt.xlim(-10,10)
# plt.show()
# exit()



output_folder="/home/ubuntu/Documents/Thesis_work/results/pwave_sig0_sig1/"

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
    if ann_name=='ecg':
        annotation=ann_name;
print("annotators for this database are: " + str(annotator_array) + " we are choosing " + str(annotation))


initial_rec_array=ws.dload_rec_names(db_name);
 
 
wo_continuation_recs=ws.rmv_continuation_rec(initial_rec_array)
rec_name_array=ws.rmv_test_rec(wo_continuation_recs)
#rec_name_array=ws.rmv_even_rec(wo_continuation_recs)
 
 
print str(rec_name_array)

#############################################################################



#dictionary to store indices of features
global_vocab_ecg={}; 
index_of_features_ecg=0;
all_features=[]

pwave_time_patient=[]# this is a list of list that will contain p wave values for all records
pwave_time_normal=[]
max_vals_patient=[] 
max_vals_normal=[]
min_vals_patient=[]
min_vals_normal=[]
var_vals_patient=[]
var_vals_normal=[]
dispersion_vals_patient=[]
dispersion_vals_normal=[]

for record in rec_name_array:
    feature_rec=[]
    rec_name=db_name+"/"+record;
    
    start_time="00:00:00"
    end_time="00:"+"30"+":00"
    #end_time="00:"+"00"+":20"
    
    signals=['0','1']
    
    for sig_num in signals: # this for loop generates feature arrays for both signals 0 and 1
        all_time_features=pecg.extract_wave_times(output_folder,record,rec_name,annotation,start_time,end_time,sig_num)
        
        #print (all_time_features)
        #exit()
        all_feature_names=['wave_dur_time','wave_ini_time','wave_ter_time','wave_asy_time','pr_on_time','pr_peak_time','pr_off_time','pp_on_time']
        stat_feature_names=[sig_num+'_'+'size_val',sig_num+'_'+'min',sig_num+'_'+'max',sig_num+'_'+'mean_val',sig_num+'_'+'var_val',sig_num+'_'+'skewness_val',sig_num+'_'+'kurtosis_val']
        #all_feature_names_s1=['s1_wave_dur_time','s1_wave_ini_time','s1_wave_ter_time','s1_wave_asy_time','s1_pr_on_time','s1_pr_peak_time','s1_pr_off_time','s1_pp_on_time']
        index_num=0;
        for arr in all_time_features:
            #print arr
            #calc stats for 1 list and make 4 features 
            #print ('index_num is: ' + str(index_num))
            size_val,min_max,mean_val,var_val,skewness_val,kurtosis_val=describe(arr)
            min_val=min_max[0]
            max_val=min_max[1]
            feature_rec.append(size_val)
            feature_rec.append(min_val)
            feature_rec.append(max_val)
            feature_rec.append(mean_val)
            feature_rec.append(var_val)
            feature_rec.append(skewness_val)
            feature_rec.append(kurtosis_val)
            
            for f_name in stat_feature_names:
                global_vocab_ecg,index_of_features_ecg=cl.fill_global_vocab(f_name+'_'+all_feature_names[index_num], index_of_features_ecg, global_vocab_ecg)
        
            index_num=index_num+1;
    
  

    
    

    
    all_features.append(feature_rec)
print(' dict is: ' + str((global_vocab_ecg)))
print("all features is: ")
print all_features
print(len(all_features))
#write to all_features to file
rw.write_value(all_features,output_folder,"all_features_readable.txt",'w')
rw.write_features_to_file(all_features,output_folder,"all_features_pickle.txt") 
rw.write_features_to_file(global_vocab_ecg,output_folder,"global_vocab_pickle.txt")
rw.write_features_to_file(rec_name_array, output_folder, "rec_name_array_pickle.txt")    
exit()
