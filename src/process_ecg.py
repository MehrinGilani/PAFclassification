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

import wfdb_setup as ws;

#variables and arrays 

def gettime(sample_num, freq, init_time):
    return float(sample_num)/float(freq)
    

def get_ecg_signal(rec_name,annotation,start_time,end_time):
    #variables and arrays 
    
    iteration=[];
    sig_time=[];
    #count=0;
    #ann_graph=[];
    #split_time0=[];
    #annotator_array=[];
    
    
    nsamp, freq, annot, init_time,sdata = ws.setupWfdb(rec_name,annotation);
  
    sig0 = [];
    sig1 = [];
    
    #physig0 is array with physical units
    physig0=[];
    physig1=[];
    
    print type(init_time);
    #print("strtim for starting value is: " + str(wfdb.strtim(init_time)));
    
    #print("total num of samples: " + str(nsamp));
    #print "Starting time of record is: "+ str(init_time);
    #print("sampling frequency is:"+ str(freq));
    

    #sample interval
    
    #required length of signal in seconds
    #num_sample_start=start_time*60*freq
    num_sample_end=end_time*60*freq
    loop_iteration=int(math.floor(num_sample_end));
    
    #print("loop iteration = " +str(loop_iteration));
    
    
    # loop runs for loop_iteration times to extract signal samples
    num_value=loop_iteration;
    
    for i in range(0,num_value):
        if wfdb.getvec(sdata.cast()) < 0:
            print "ERROR: getvec() < 0";
            exit();
        else:
            #signal values in adu units:
            sig0.append(sdata[0]);
            sig1.append(sdata[1]);
            
            sig_time.append(gettime(i, freq, init_time));
            #print("time for sample " + str(i) + "is: " + str(sig_time[i]));
            #convert adu units to physical units and save in physig0 and 1 (later generalise it for n number of signals)
            physig0.append(aduphys(0,sig0[i]));
            physig1.append(aduphys(1,sig1[i]));
           
            #append iteration number as value in 
            iteration.append(i);
            
        
    #getann reads next annotation and returns 0 when successful
    while wfdb.getann(0,annot) ==0:
        if annot.time>num_value:
            #print("annot.time>number of samples extracted");
            break;
        #  annot.time is time of the annotation, in samples from the beginning of the record.
        print wfdb.timstr(-annot.time),"(" + str(annot.time)+ ")",wfdb.annstr(annot.anntyp), annot.subtyp,annot.chan, annot.num
        print ("signal value at this annotation is : " + str(physig0[annot.time])+" "+ str(sig_time[annot.time]));
    
    
    return(physig0,physig1,sig_time)    
    
def extract_pwave(output_folder,record,rec_name,annotation,start_time,end_time):
    ## time for which you want to read record ##
    nsamp, freq, annot, init_time,sdata=ws.setupWfdb(rec_name, annotation)
    
    
    #print("nsamp:"+str(nsamp)+" freq:"+str(freq));
    
    rdann_file=output_folder+"ecgpu_output.txt"
    output_ann="output_annotator"
    
    
    #ecgpu_path="/home/ubuntu/Documents/Thesis_work/ecgpuwave-1.3.2"
    #change directory to output_folder
    os.chdir(output_folder) 
    
   
    os.system("rm -f "+rdann_file)
    cmd_create_ann="ecgpuwave -r "+rec_name +" "+"-a"+" "+output_ann+ " -f "+start_time+" -t "+ end_time +" -i "+annotation
    print(cmd_create_ann)
    os.system(cmd_create_ann)
    
    #use rdann to ouput annotations as a text file
    cmd_disp_ann="rdann -r "+rec_name+" -a output_annot"
    #push output text to a file 
    print (cmd_disp_ann)
    os.system(cmd_disp_ann +">" +rdann_file)

    l_p_wave_times=[] # this will contain p wave values for 1 rec and will be emptied everytime
    
    ## code for p wave time calculation goes here ####
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
            l_p_wave_times.append(p_duration_ms)
     

    return l_p_wave_times

def separate_p_n_pwave(record,db_name,p_wave_times,pwave_time_patient,pwave_time_normal):
    if ('p' in record) or (db_name == 'afdb'):
        pwave_time_patient.append(p_wave_times) # this would be a list of list of p_wave_times_of all_recs 
        
        #print ("pwave_time_patient is : " +str(pwave_time_patient))
    elif ('n' in record) or (db_name == 'nsrdb'):
        pwave_time_normal.append(p_wave_times)
    
    return pwave_time_patient,pwave_time_normal

def calc_pwave_max(p_wave_times):
    #this func will take p wave array for 1 patient and calculate and return the max value in pwave  
    pwave_max=np.max(p_wave_times)
    return pwave_max
    
def calc_pwave_var(p_wave_times):  
    variance=np.var(p_wave_times)
    return variance
    
def calc_pwave_disp(p_wave_times):
    max_val=np.max(p_wave_times)
    min_val=np.min(p_wave_times)
    disp_val=max_val-min_val
    return disp_val


    
    
    
    
    