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
    

