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



output_folder="/home/ubuntu/Documents/Thesis_work/results/thesis_images/chapter_5/"

db_name="afpdb";
initial_rec_array=[];
rec_name_array=[];
annotator_array=[];
wo_continuation_recs=[]
 
#recs_to_remove=['n24','n27','n28'];
 
annotator_array=ws.dload_annotator_names(db_name);
for ann_name in annotator_array:
    print ann_name
    if ann_name == "atr":
        annotation=ann_name;
    elif ann_name == "qrs":
        annotation=ann_name;
    elif ann_name=='ecg':
        annotation=ann_name;

# if db_name == 'afdb':
#     annotation='atr'
print("annotators for this database are: " + str(annotator_array) + " we are choosing " + str(annotation))


initial_rec_array=ws.dload_rec_names(db_name);
 
 
wo_continuation_recs=ws.rmv_continuation_rec(initial_rec_array)
rec_name_array=ws.rmv_test_rec(wo_continuation_recs)
#rec_name_array=ws.rmv_even_rec(wo_continuation_recs)
 
 
print str(rec_name_array)



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
 
 
#wo_continuation_recs=ws.rmv_continuation_rec(initial_rec_array)
rec_name_array=ws.rmv_test_rec(initial_rec_array)
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
    signals=['0','1']
    for sig_num in signals: # this for loop generates feature arrays for both signals 0 and 1
        all_time_features=pecg.extract_wave_times(output_folder,record,rec_name,annotation,start_time,end_time,sig_num)
        
#         all_feature_names=['wave_dur_time','wave_ini_time','wave_ter_time','wave_asy_time','pr_on_time','pr_peak_time','pr_off_time']
#         stat_feature_names=[sig_num+'_'+'size_val',sig_num+'_'+'min',sig_num+'_'+'max',sig_num+'_'+'mean_val',sig_num+'_'+'var_val',sig_num+'_'+'skewness_val',sig_num+'_'+'kurtosis_val']
        all_feature_names=['wave_dur_time']
        stat_feature_names=[sig_num+'_'+'size_val',sig_num+'_'+'min',sig_num+'_'+'max',sig_num+'_'+'mean_val',sig_num+'_'+'var_val',sig_num+'_'+'skewness_val',sig_num+'_'+'kurtosis_val']
        
        #all_feature_names_s1=['s1_wave_dur_time','s1_wave_ini_time','s1_wave_ter_time','s1_wave_asy_time','s1_pr_on_time','s1_pr_peak_time','s1_pr_off_time','s1_pp_on_time']
        index_num=0;
        for arr in all_time_features:
            index_num=index_num+1;
            trend_s0=pecg.extract_trend_frm_list(arr)
            #trend_s1=pecg.extract_trend_frm_list(s1_wave_ini_time)
            ###plot pwave points  and save fig ####
            fig_pwave,plot_pwave=graphs.plotScatter(record,range(len(trend_s0)),trend_s0, "Number of 10 P wave-length intervals", " P dur variability (samples) ", "P dur variability", 'b',xlim_lo=0, xlim_hi=140, ylim_lo=0, ylim_hi=140,axline=0)
            plt.plot(range(len(trend_s0)), np.poly1d(np.polyfit(range(len(trend_s0)), trend_s0, 1))(range(len(trend_s0))))
            plt.xlim(0,100);
            plt.ylim(0,140);
            #fig_pwave,plot_pwave=graphs.plot_simple(rec_name,range(len(p_wave_times_0)),p_wave_times_0, "serial num", " pwave duration (ms) ", "s0 p wave duration ", 'g',xlim_lo=0, xlim_hi=0, ylim_lo=0, ylim_hi=0)
            fig_pwave.savefig(output_folder+"trend_s0_pwave_ini_"+record+".pdf",format='pdf')
            plt.show()   
            print("done")
            exit()
    #all_features.append(feature_rec)
    


    #pwave_time_patient,pwave_time_normal=pecg.separate_p_n_pwave(record,db_name,p_wave_times,pwave_time_patient,pwave_time_normal) #i dont thing we need this
#     feature_rec.append(p_wave_times_0)
#     global_vocab_ecg,index_of_features_ecg=cl.fill_global_vocab("pwave_time_0", index_of_features_ecg, global_vocab_ecg)
    
#     ## this would not work for now
#     pwave_max_0,pwave_max_1=pecg.calc_pwave_max(p_wave_times_0,p_wave_times_1)
#     feature_rec.append(pwave_max_0)
#     global_vocab_ecg,index_of_features_ecg=cl.fill_global_vocab("pwave_max_0", index_of_features_ecg, global_vocab_ecg)
#     output_folder=
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
