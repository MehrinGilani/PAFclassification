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


output_folder="/home/ubuntu/Documents/Thesis_work/testing/"


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

### manually assigning annotation value
#annotation="qrs"
initial_rec_array=ws.dload_rec_names(db_name);
 
 
wo_continuation_recs=ws.rmv_continuation_rec(initial_rec_array)
rec_name_array=ws.rmv_test_rec(wo_continuation_recs)
#rec_name_array=ws.rmv_even_rec(wo_continuation_recs)
 
 
print str(rec_name_array)

#############################################################################



#dictionary to store indices of features
global_vocab={}; 
index_of_features=0;
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
time_interval=5;
for record in rec_name_array:
    
    rec_name=db_name+"/"+record;
    #rec_name=record
    ## time for which you want to read record ##
    nsamp, freq, annot, init_time,sdata=ws.setupWfdb(rec_name, annotation)
    
    
    #print("nsamp:"+str(nsamp)+" freq:"+str(freq));
    
    rdann_file=output_folder+"ecgpu_output.txt"
    output_ann="output_annotator"
    
    start_time="00:00:00"
    end_time="00:"+"30"+":00"

    
    #ecgpu_path="/home/ubuntu/Documents/Thesis_work/ecgpuwave-1.3.2"
    #change directory to output_folder
    os.chdir(output_folder) 
    
    #ecgpuwave -r afpdb/p02 -a output_annotator -f 00:25:00 -t 00:30:00
    os.system("rm -f "+rdann_file)
    cmd_create_ann="ecgpuwave -r "+rec_name +" "+"-a"+" "+output_ann+ " -f "+start_time+" -t "+ end_time +" -i "+annotation
    print(cmd_create_ann)
    os.system(cmd_create_ann)
    
    #use rdann to ouput annotations as a text file
    cmd_disp_ann="rdann -r "+rec_name+" -a output_annot"
    #push output text to a file 
    print (cmd_disp_ann)
    os.system(cmd_disp_ann +">" +rdann_file)

    p_wave_times=[] # this will contain p wave values for 1 rec and will be emptied everytime
    
    ## code for p wave goes here ####
    f=open(rdann_file,'r')
    for line in f:
        
        temp=line.split()
        if (temp[2] == '(') and (temp[-1] =='0'):
            start_sample_num=float(temp[1])
            #print ("start sample num is : "+ str(start_sample_num))
        elif (temp[2] == ')') and (temp[-1] =='0'):
            end_sample_num=float(temp[1])
            #print("end_sample_num is: " +str(end_sample_num))
            p_duration_ms=(end_sample_num-start_sample_num) *(1000/freq);
            p_wave_times.append(p_duration_ms)
     
    if 'p' in record:
        pwave_time_patient.append(p_wave_times) # this would be a list of list of p_wave_times_of all_recs 
        #print ("pwave_time_patient is : " +str(pwave_time_patient))
    elif 'n' in record:
        pwave_time_normal.append(p_wave_times)
        #print ("pwave_time_normal is : " +str(pwave_time_normal))

#     time_interval=5;
#     start_time="00:00:00"
#     end_time="00:05:00"

#print pwave_time_patient

rw.write_features_to_file(pwave_time_patient,output_folder,"pwave_time_patient_pickle.txt") 
rw.write_features_to_file(pwave_time_normal,output_folder,"pwave_time_normal_pickle.txt") 

for arr in pwave_time_patient:
    max_pwave_rec=np.max(arr)
    min_pwave_rec=np.min(arr)
    var_pwave_rec=np.var(arr)
     
    max_vals_patient.append(max_pwave_rec)
    min_vals_patient.append(min_pwave_rec)
    var_vals_patient.append(var_pwave_rec)
 
for max_val,min_val in zip(max_vals_patient,min_vals_patient):
    dispersion_vals_patient.append(max_val-min_val)


for arr in pwave_time_normal:
    max_pwave_rec=np.max(arr)
    min_pwave_rec=np.min(arr)
    var_pwave_rec=np.var(arr)
     
    max_vals_normal.append(max_pwave_rec)
    min_vals_normal.append(min_pwave_rec)
    var_vals_normal.append(var_pwave_rec)
 
for max_val,min_val in zip(max_vals_normal,min_vals_normal):
    dispersion_vals_normal.append(max_val-min_val)


mean_pmax_patient=np.mean(max_vals_patient)
mean_pmax_normal=np.mean(max_vals_normal)


mean_pdisp_patient=np.mean(dispersion_vals_patient)
mean_pdisp_normal=np.mean(dispersion_vals_normal)

mean_pvar_patient=np.mean(var_vals_patient)
mean_pvar_normal=np.mean(var_vals_normal)


#########3 save to csv #############3

np.savetxt(output_folder+"max_vals_patient.csv",max_vals_patient,fmt="%s",delimiter=',',newline='\n')
np.savetxt(output_folder+"max_vals_normal.csv",max_vals_normal,fmt="%s",delimiter=',',newline='\n')

np.savetxt(output_folder+"dispersion_vals_patient",dispersion_vals_patient,fmt="%s",delimiter=',',newline='\n')
np.savetxt(output_folder+"dispersion_vals_normal",dispersion_vals_normal,fmt="%s",delimiter=',',newline='\n')

np.savetxt(output_folder+"var_vals_patient",var_vals_patient,fmt="%s",delimiter=',',newline='\n')
np.savetxt(output_folder+"var_vals_normal",var_vals_normal,fmt="%s",delimiter=',',newline='\n')




#print("avg value of pmax patient is : " + str(mean_pmax_patient))
print("avg_value of pmax normal is : " + str(mean_pmax_normal))

#print("avg value of pdisp patient is : " + str(mean_pdisp_patient))
print("avg_value of pdisp normal is : " + str(mean_pdisp_normal))

#print("avg value of pvar patient is : " + str(mean_pvar_patient))
print("avg_value of pvar normal is : " + str(mean_pvar_normal))
    
    #####
    
    
#     annotation=output_ann
#     nsamp, freq, annot, init_time,sdata = ws.setupWfdb(rec_name,annotation);
#     
#     num_sample_end=end_time*60*128
#     loop_iteration=int(math.floor(num_sample_end));
#     
#     num_value=loop_iteration;
#     #getann reads next annotation and returns 0 when successful
#     while wfdb.getann(0,annot) ==0:
#         print("in while")
#         if annot.time>num_value:
#             #print("annot.time>number of samples extracted");
#             break;
#         #  annot.time is time of the annotation, in samples from the beginning of the record.
#         print wfdb.timstr(-annot.time),"(" + str(annot.time)+ ")",wfdb.annstr(annot.anntyp), annot.subtyp,annot.chan, annot.num
#         #print ("signal value at this annotation is : " + str(physig0[annot.time])+" "+ str(sig_time[annot.time]));
#     
    
#     ################################3
#     start_time=0;
#     end_time=30; #time in mins
#     physig0,physig1,sig_time=pecg.get_ecg_signal(rec_name,annotation,start_time,end_time)
#     #starting time of record
#     print("starting time of record is: " + (wfdb.timstr(0L)));
#     ############################################
#     #print array to check
#     #print("aduphys for sig0: " ,physig0);
#     #print("aduphys for sig1: " ,physig1);
#     
#     ##Plot graph
#     #fig = plt.figure()
#     
#     ############################################
#     #write signal value to file
#     f=open(output_folder+"ecg_values.txt",'w')
#     for i in physig0:
#         f.write(str(i))
#         f.write("\n")
#     
#     cmd="fft ecg_values.txt >  "+output_folder+"fft.txt"
#     os.system(cmd)
#     ############################################
#     #read fft file
#     f=open(output_folder+"fft.txt",'r');
#     fft_values=[]
#     for i in f:
#         fft_values.append(i)
#     
#     ############################################
#     #y values physical units
    #plt.figure()
    #plt.plot(sig_time,physig0,linestyle="-");    
    #plt.plot(sig_time,physig1,linestyle="-");
    
    
    #axis labels and legends
    #plt.xlabel("time elapsed from start of record");
    #plt.ylabel(" ecg (mV) ");
    #plt.title(" %s" % rec_name);
    
    ##ax = fig.gca();
    #ax.set_xticks(np.arange(0,(num_value/freq),0.2));
    #ax.set_yticks(np.arange(sig_min,sig_max,0.5));
    
    #keep grid
    #ax.grid(True);
    #ax.set_xticklabels([])
    
    
    ####plot window std_dev and save fig ####
    #fig_std,plot_std=graphs.plot_simple(rec_name,range(len(fft_values)), fft_values, "sample number? ", "fft", "fft", "r", 0, 0,0, 0)
    #fig_std.savefig(output_folder+"fft_"+record+".png")



    ##print p wave annotations
    



#     plt.figure()
#     plt.plot(fft_values)
#     plt.xlabel("point number");
#     plt.ylabel(" fft ");
#     plt.title(" %s" % rec_name);
#     
#     plt.show();





