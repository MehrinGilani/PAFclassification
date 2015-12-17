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


def get_RR_interval(rec_name,annotation,start_time,end_time):
        #setup wfdb (change annotator here)
    rr_int=0;
    t=0;
    beats=0;
    sig_time=[];
    
    #change annotations
    nsamp, freq, annot, init_time = ws.setupWfdb(rec_name,annotation);
    RR_sec_func=[];
    #num_sample_end=38400; # this translates to 230339 annot time which means 30 mins of data ## check this
    num_sample_start=start_time*60*freq
    num_sample_end=end_time*60*freq
    ###### check if its the same record or new. if new , shift annot.time

#         annot.time
    
    #getann reads next annotation and returns 0 when successful
    # annot.time is time of the annotation, in samples from the beginning of the record.
    #when getting samples for cross-validation and testing, just adjust annot.time so it starts reading samples from there
    
    
    t = annot.time;
    ##comment june 16
    #print ("annot.time at the beginning is: " + str(annot.time))
    
    #annot_file=open('/home/ubuntu/Documents/eclispe_workspace/test_one/my_first_pyproj/research_March17/annot_type' +str(record)+'.txt','a');
    
    while wfdb.getann(0, annot) == 0:
        if annot.time>num_sample_start and annot.time<num_sample_end:
            #code for extracting time:
            time=wfdb.timstr(-annot.time)
            split_time=time.split(":")
            hr_time=split_time[0];
            minute_time=split_time[1];
            if len(split_time) >2:
                sec_time=split_time[2];
            
    
            if wfdb.wfdb_isqrs(annot.anntyp):
    #             if same_as_prev[record] == 1 and int(hr_time) >= 1 and int(hr_time) < 23: 
                    #annot.time=annot.time + prev_annot_time;
                rr_int = annot.time - t
                beats=beats+1;
                rr_sec=rr_int/freq
                rr_sec_rounded=round(rr_sec,3);
                
                ###############testing here############################
         
                
                ############################################
                
                RR_sec_func.append(rr_sec_rounded);
                # sampling intervals (e.g., if the original recording was sampled at 128 samples per second, then an 
                t = annot.time
                #print ("annot.time after rr interval is: " + str(t))
                
                #print ("-annot.time after rr interval is: " +  wfdb.timstr(-annot.time))
                #print ("annot.time in if is: " + str(annot.time))
                #prev_annot_time.append(annot.time)
        
    
    #print ("beats = "+ str(beats))
    return RR_sec_func

def get_delta_rr(RR_sec):
    #####Extract DELTA RR intervals here #########
    delta_RR_sec = [];
    for i in range(len(RR_sec)):
        if i<len(RR_sec)-1:
            delta_RR_value=RR_sec[i+1]-RR_sec[i]
            delta_RR_sec.append(delta_RR_value)
    return delta_RR_sec;



def get_window_std_dev(rr_array,window_size):
    #rr_sec_window=[];
    std_dev_array=[];
    i=0;
    std_dev_val=0;
    while (i+window_size-1) < (len(rr_array)):
        #rr_sec_window=list(rr_array[i:i+window_size-1]);
        std_dev_val=np.std(rr_array[i:i+window_size-1]);
        std_dev_array.append(std_dev_val);
        i=i+window_size;
    return std_dev_array;


# def get_short_term_hrv(feature,rec_name,annotation,start_time,end_time,output_folder):
#     get_hrv_path="/home/ubuntu/mehrin/wfdb/wfdb-10.5.23/HRV"
#     feature_file_path=output_folder+"feature_file.txt"
#     feature="\""+feature+"\""
#     
#     os.chdir(get_hrv_path) 
# 
#     cmd="./get_hrv -M "+rec_name +" "+ annotation+ " "+ start_time +" "+ end_time +" "+ "| " +"egrep " + feature +" > " + feature_file_path
#     #print "command to be executed is:" + cmd
#     os.system(cmd)
#     
#     ##read feature from file into list
#     
#     f=open(feature_file_path,'r')
#     for line in f:
#         temp=line.split()
#         feature_value=temp[-1]
#         if "nan" in feature_value:
#             print ("nan" + str(feature))
#             feature_value=-1;
def get_short_term_hrv(feature_list,rec_name,annotation,start_time,end_time,output_folder):    
    list_of_list_30min_features=[] # each list in this list of lists is for one 5min interval and contains tuples with feature names and vlaues
    # everytime the loop runs it extracts feauters for those 5 mins
    feature_name_val=[]
    print("calculating 30 min hrv features from toolkit")
    print ("start time: " + str(start_time))
    print ("end time: " + str(end_time))
    #feature_list="'SDNN|AVNN|rMSSD|pNN50|TOT PWR|VLF PWR|LF PWR|HF PWR|LF/HF'"
    all_file=output_folder+"hrv_temp_30min.txt"
    get_hrv_path="/home/ubuntu/mehrin/wfdb/wfdb-10.5.23/HRV"
    os.chdir(get_hrv_path) 

    cmd="./get_hrv -M "+rec_name +" "+ annotation+ " "+ start_time +" "+ end_time +" "+ "| " +"egrep " + feature_list +" > " + all_file
    os.system(cmd)
    
    ####### pick up features from all.txt and put in separate files ########
    #feature_array= ['SDNN','AVNN','rMSSD' ,'pNN50','TOT PWR','VLF PWR','LF PWR','HF PWR','LF/HF']   
    i=0;
    f=open(all_file,'r')
    for line in f:
        temp=line.split()
        #make feature_tuple 
        feature_tuple=(str(temp[0])+"_"+"30min",float(temp[-1]))
        feature_name_val.append(feature_tuple)
        #print (feature_tuple)
        #print((feature_name_val[0])[0])
        #print((feature_name_val[0])[1])
        #print(type((feature_name_val[0])[1]))
        #exit()
        i=i+1
    
    list_of_list_30min_features.append(feature_name_val)
    #print ("5min feature names and values are: " +str(feature_name_val))
    
    
#print ("list of list for 5 min features is: " +str(list_of_list_5min_features))  
    return list_of_list_30min_features  

def get_5min_hrv_features(rec_name,annotation,total_min,output_folder):
    start_min_arr=range(0,total_min,5)
    end_min_arr=range(5,total_min+5,5)
    list_of_list_5min_features=[] # each list in this list of lists is for one 5min interval and contains tuples with feature names and vlaues
    # everytime the loop runs it extracts feauters for those 5 mins
    num_5min_interval=1
    for start_min, end_min in zip(start_min_arr,end_min_arr): 
        start_time="00:"+str(start_min)+":00"
        end_time="00:"+str(end_min)+":00"
        feature_name_val=[]
        
        #keeps track of which chunck of 5 min interval it is
        print("calculating 5 min hrv features from toolkit")
        print ("start time: " + str(start_time))
        print ("end time: " + str(end_time))
        feature_list="'SDNN|AVNN|rMSSD|pNN50|TOT PWR|VLF PWR|LF PWR|HF PWR|LF/HF'"
        all_file=output_folder+"all.txt"
        get_hrv_path="/home/ubuntu/mehrin/wfdb/wfdb-10.5.23/HRV"
        os.chdir(get_hrv_path)
        cmd="./get_hrv -M "+rec_name +" "+ annotation+ " "+ start_time +" "+ end_time +" "+ "| " +"egrep " + feature_list +" > " + all_file
        os.system(cmd)
        
        ####### pick up features from all.txt and put in separate files ########
        #feature_array= ['SDNN','AVNN','rMSSD' ,'pNN50','TOT PWR','VLF PWR','LF PWR','HF PWR','LF/HF']   
        i=0;
        f=open(all_file,'r')
        for line in f:
            temp=line.split()
            #make feature_tuple 
            feature_tuple=(str(temp[0])+"_"+str(num_5min_interval),float(temp[-1]))
            feature_name_val.append(feature_tuple)
            #print (feature_tuple)
            #print((feature_name_val[0])[0])
            #print((feature_name_val[0])[1])
            #print(type((feature_name_val[0])[1]))
            #exit()
            i=i+1
        
        list_of_list_5min_features.append(feature_name_val)
        num_5min_interval=num_5min_interval+1
        #print ("5min feature names and values are: " +str(feature_name_val))
        
        
    #print ("list of list for 5 min features is: " +str(list_of_list_5min_features))  
    return list_of_list_5min_features

 


## add get_sodp measure function here