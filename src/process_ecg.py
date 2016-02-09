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

def extract_wave_times(output_folder,record,rec_name,annotation,start_time,end_time,signal_num):   
    ##The num field of each WFON and WFOFF annotation designates the type of waveform with which it is associated: 0 for a P wave, 1 for a QRS complex, or 2 for a T wave. 
    

#     ## doing wave num encoding here:
#     if wave_name == 'p':
#         wave_num = '0'
#     elif wave_name == 'r':
#         wave_num = '1'
#     elif wave_name =='t':
#         wave_num = '2'
    
    ## time for which you want to read record ##
    nsamp, freq, annot, init_time,sdata=ws.setupWfdb(rec_name, annotation)
    
    
    rdann_file=output_folder+"ecgpu_output.txt"
    output_ann="output_annotator"
    
    os.chdir(output_folder) 
    
    signal=signal_num;
    os.system("rm -f "+rdann_file)
    cmd_create_ann="ecgpuwave -r "+rec_name +" "+"-a"+" "+output_ann+ " -f "+start_time+" -t "+ end_time +" -i "+annotation+" -s "+signal
    print(cmd_create_ann)
    os.system(cmd_create_ann)
    
    #use rdann to ouput annotations as a text file
    cmd_disp_ann="rdann -r "+rec_name+" -a output_annot"
    #push output text to a file 
    print (cmd_disp_ann)
    os.system(cmd_disp_ann +">" +rdann_file)
    
    ## features we want to extract:
    # p_dur=p_off-p_on
    # p_ini=p_p_peak-p_on
    # p_ter = p_off - p_peak
    # p_asy=p_ter/p_ini
    
    #pr_on
    #pr_peak
    #r_off
   
    l_wave_dur_time=[]  # this will contain p/t wave values for 1 rec and will be emptied everytime
    l_wave_ini_time=[]
    l_wave_ter_time=[]
    l_wave_asy_time=[]
    l_pr_on_time=[]
    l_pr_peak_time=[]
    l_pr_off_time=[]
    l_pp_on_time=[]
    
    ## code for p wave time calculation goes here ####
    f=open(rdann_file,'r')
    p_true=0
    for line in f:
        
        temp=line.split()
        if (temp[2] == '(') and (temp[-1] =='0'):
            start_sample_num=float(temp[1])
            #print ("start sample num is : "+ str(start_sample_num))
        elif (temp[2] == 'p') and (temp[-1] =='0'):
            peak_sample_num=float(temp[1])
            p_true=1;
        elif (temp[2] == ')') and (temp[-1] =='0'):
            end_sample_num=float(temp[1])
            #print("end_sample_num is: " +str(end_sample_num))
            wave_duration_ms=(end_sample_num-start_sample_num) *(1000/freq);
            wave_ini_ms=(peak_sample_num-start_sample_num)*(1000/freq)
            wave_ter_ms=(end_sample_num-peak_sample_num)*(1000/freq)
            wave_asy_ms=wave_ter_ms/wave_ini_ms
            l_wave_dur_time.append(wave_duration_ms)
            l_wave_ini_time.append(wave_ini_ms)
            l_wave_ter_time.append(wave_ter_ms)
            l_wave_asy_time.append(wave_asy_ms)
        if (temp[2] == 'N') and (temp[-1] =='0') and (p_true== 1): # this is looking for r peakvand making sure that p wave was calculated previously
            r_peak_sample=float(temp[1])
            pr_on_ms=(r_peak_sample -start_sample_num)*(1000/freq)
            pr_peak_ms=(r_peak_sample-peak_sample_num)*(1000/freq)
            pr_off_ms=(r_peak_sample-end_sample_num)*(1000/freq)
            p_true=0;
            l_pr_on_time.append(pr_on_ms)
            l_pr_peak_time.append(pr_peak_ms)
            l_pr_off_time.append(pr_off_ms)
                    
    for i in range(len(l_wave_dur_time)-1):
        pp_on_ms=l_wave_dur_time[i+1]- l_wave_dur_time[i]
        l_pp_on_time.append(pp_on_ms)
    #print ("pwave duration from signal 0 is : " + str(p_duration_ms))
    ##### extract p wave times for second signal 
    all_time_features=[l_wave_dur_time,l_wave_ini_time,l_wave_ter_time,l_wave_asy_time,l_pr_on_time,l_pr_peak_time,l_pr_off_time]
    return all_time_features

def extract_pwave(output_folder,record,rec_name,annotation,start_time,end_time):
    ## time for which you want to read record ##
    nsamp, freq, annot, init_time,sdata=ws.setupWfdb(rec_name, annotation)
    
    
    #print("nsamp:"+str(nsamp)+" freq:"+str(freq));
    
    rdann_file=output_folder+"ecgpu_output.txt"
    s1_rdann_file=output_folder+"s1_ecgpu_output.txt"
    output_ann="output_annotator"
    
    
    #ecgpu_path="/home/ubuntu/Documents/Thesis_work/ecgpuwave-1.3.2"
    #change directory to output_folder
    os.chdir(output_folder) 
    
    signal_0="0";
    os.system("rm -f "+rdann_file)
    cmd_create_ann="ecgpuwave -r "+rec_name +" "+"-a"+" "+output_ann+ " -f "+start_time+" -t "+ end_time +" -i "+annotation+" -s "+signal_0
    print(cmd_create_ann)
    os.system(cmd_create_ann)
    
    #use rdann to ouput annotations as a text file
    cmd_disp_ann="rdann -r "+rec_name+" -a output_annot"
    #push output text to a file 
    print (cmd_disp_ann)
    os.system(cmd_disp_ann +">" +rdann_file)

    s0_l_p_wave_times=[] # this will contain p wave values for 1 rec and will be emptied everytime
    
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
            s0_l_p_wave_times.append(p_duration_ms)
            #print ("pwave duration from signal 0 is : " + str(p_duration_ms))
    ##### extract p wave times for second signal
    signal_1="1";
    os.system("rm -f "+s1_rdann_file)
    cmd_create_ann="ecgpuwave -r "+rec_name +" "+"-a"+" "+output_ann+ " -f "+start_time+" -t "+ end_time +" -i "+annotation+" -s "+signal_1
    print(cmd_create_ann)
    os.system(cmd_create_ann)
     
    #use rdann to ouput annotations as a text file
    cmd_disp_ann="rdann -r "+rec_name+" -a output_annot"
    #push output text to a file 
    print (cmd_disp_ann)
    os.system(cmd_disp_ann +">" +s1_rdann_file)
 
    s1_l_p_wave_times=[] # this will contain p wave values for 1 rec and will be emptied everytime
     
    ## code for p wave time calculation goes here ####
    f=open(s1_rdann_file,'r')
    for line in f:
         
        temp=line.split()
        if (temp[2] == '(') and (temp[-1] =='0'):
            start_sample_num=float(temp[1])
            #print ("start sample num is : "+ str(start_sample_num))
        elif (temp[2] == ')') and (temp[-1] =='0'):
            end_sample_num=float(temp[1])
            #print("end_sample_num is: " +str(end_sample_num))
            p_duration_ms=(end_sample_num-start_sample_num) *(1000/freq);
            #print ("pwave duration from signal 1 is : " + str(p_duration_ms))
            s1_l_p_wave_times.append(p_duration_ms)
    return s0_l_p_wave_times,s1_l_p_wave_times
    #return s0_l_p_wave_times

def separate_p_n_pwave(record,db_name,p_wave_times,pwave_time_patient,pwave_time_normal):
    if ('p' in record) or (db_name == 'afdb'):
        pwave_time_patient.append(p_wave_times) # this would be a list of list of p_wave_times_of all_recs 
        
        #print ("pwave_time_patient is : " +str(pwave_time_patient))
    elif ('n' in record) or (db_name == 'nsrdb'):
        pwave_time_normal.append(p_wave_times)
    
    return pwave_time_patient,pwave_time_normal

def calc_pwave_max(pwave_times_0,pwave_times_1):
    #this func will take p wave array for 1 patient and calculate and return the max value in pwave  
    if (len(pwave_times_0) != 0):
        pwave_max_0=np.max(pwave_times_0)
    else:
        pwave_max_0=-1
    if (len(pwave_times_1) !=0):
        pwave_max_1=np.max(pwave_times_1)
    else:
        pwave_max_1=-1
    
    return pwave_max_0,pwave_max_1

def calc_pwave_var(pwave_times_0,pwave_times_1):
    if (len(pwave_times_0) != 0):
        variance_0=np.var(pwave_times_0)
    else: 
        variance_0 = -1
    if (len(pwave_times_1) !=0):
        variance_1=np.var(pwave_times_1)
    else:
        variance_1=-1
    return variance_0,variance_1
    
def calc_pwave_disp(pwave_times_0,pwave_times_1):
    if (len(pwave_times_0) != 0):
        max_val_0=np.max(pwave_times_0)
        min_val_0=np.min(pwave_times_0)
        disp_val_0=max_val_0-min_val_0
    else:
        disp_val_0=-1
    if (len(pwave_times_1) !=0):
        max_val_1=np.max(pwave_times_1)
        min_val_1=np.min(pwave_times_1)
        disp_val_1=max_val_1-min_val_1
    else: 
        disp_val_1=-1
    
    return disp_val_0,disp_val_1

def extract_trend_frm_list(wave_times):
    wave_variability=[]
    counter=0;
    wave_segment=[]
    for val in wave_times:
        wave_segment.append(val)
        counter+=1;
        if counter == 10:
            #compute pwave variability value
            percentile_90=np.percentile(wave_segment,90)
            percentile_10=np.percentile(wave_segment,10)
            diff=percentile_90-percentile_10
            wave_variability.append(diff)
            counter=0
            wave_segment[:]=[]
    return wave_variability
    
    
    
    
    