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

import process_ecg as pecg


output_folder="/home/ubuntu/Documents/Thesis_work/testing/pwave_test/pwave_code_debug/challenge_db_pwave/raw_pwave_analysis/"

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
    
    #p_wave_times=[] # this will contain p wave values for 1 rec and will be emptied everytime
    p_wave_times_0,p_wave_times_1=pecg.extract_pwave(output_folder,record,rec_name,annotation,start_time,end_time)
    
    #pwave_time_patient,pwave_time_normal=pecg.separate_p_n_pwave(record,db_name,p_wave_times,pwave_time_patient,pwave_time_normal) #i dont thing we need this
    feature_rec.append(p_wave_times_0)
    global_vocab_ecg,index_of_features_ecg=cl.fill_global_vocab("pwave_time_0", index_of_features_ecg, global_vocab_ecg)
    
#     ## this would not work for now
#     pwave_max_0,pwave_max_1=pecg.calc_pwave_max(p_wave_times_0,p_wave_times_1)
#     feature_rec.append(pwave_max_0)
#     global_vocab_ecg,index_of_features_ecg=cl.fill_global_vocab("pwave_max_0", index_of_features_ecg, global_vocab_ecg)
#     
#     feature_rec.append(pwave_max_1)
#     global_vocab_ecg,index_of_features_ecg=cl.fill_global_vocab("pwave_max_1", index_of_features_ecg, global_vocab_ecg)
#     
#     pwave_var_0,pwave_var_1=pecg.calc_pwave_var(p_wave_times_0,p_wave_times_1)
#     feature_rec.append(pwave_var_0)
#     global_vocab_ecg,index_of_features_ecg=cl.fill_global_vocab("pwave_var_0", index_of_features_ecg, global_vocab_ecg)
#     
#     feature_rec.append(pwave_var_1)
#     global_vocab_ecg,index_of_features_ecg=cl.fill_global_vocab("pwave_var_1", index_of_features_ecg, global_vocab_ecg)
#      
#      
#     pwave_disp_0,pwave_disp_1=pecg.calc_pwave_disp(p_wave_times_0,p_wave_times_1)
# 
#          
#     feature_rec.append(pwave_disp_0)
#     global_vocab_ecg,index_of_features_ecg=cl.fill_global_vocab("pwave_disp_0", index_of_features_ecg, global_vocab_ecg)
#     
#     feature_rec.append(pwave_disp_1)
#     global_vocab_ecg,index_of_features_ecg=cl.fill_global_vocab("pwave_disp_1", index_of_features_ecg, global_vocab_ecg)
#     
    
    all_features.append(feature_rec)
print("all features is: ")
print all_features
print(len(all_features))
#write to all_features to file
rw.write_value(all_features,output_folder,"all_features_readable.txt",'w')
rw.write_features_to_file(all_features,output_folder,"all_features_pickle.txt") 
rw.write_features_to_file(global_vocab_ecg,output_folder,"global_vocab_pickle.txt")
rw.write_features_to_file(rec_name_array, output_folder, "rec_name_array_pickle.txt")    
exit()
