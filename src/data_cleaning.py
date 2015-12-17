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

def gettime(sample_num, freq, init_time):
        time_sec=float(sample_num)/float(freq);
        time_min=time_sec/60.0;
        time_hour=time_min/60.0;
        return time_hour
    
#this function uses quotient filter to filter ectopic beats in RR_interval array
def quotient_filt(unclean_array):
    clean_array=[];
    index=[];
    for i in range(0,len(unclean_array)-1):
        rule_1=unclean_array[i]/unclean_array[i+1];
        rule_2=unclean_array[i+1]/unclean_array[i];
        if (rule_1 >= 1.2 or rule_1<=0.8 or rule_2>=1.2 or rule_2<=0.8):
            index.append(i); #if above condition is true, append n value to index
    
    #copy entire unclean_array array into clean array
    clean_array=list(unclean_array);
    #print (" unclean_array_clean is: " +str(unclean_array_clean));
    #print (len(index));
    
    
    ##remove the uncleaned values
    for i in sorted(index, reverse=True):
        del clean_array[i]
        #del clean_array[i+1]
        #del clean_array[i+2]
    #print ("length of clean_array after quotient filter is: " + str(len(clean_array)));
    return clean_array

def square_filt(unclean_array2):
    clean_array2=[];
    index2=[];
    for i in range(0,len(unclean_array2)):
        #if(unclean_array2[i] <0.3 or unclean_array2[i] >5):
        if(unclean_array2[i] >5):
            index2.append(i); #if above condition is true, append n value to index
    clean_array2=list(unclean_array2);
    ##remove the uncleaned values
    for i in sorted(index2, reverse=True):
        del clean_array2[i];
    #print ("length of clean_array after square filter is: " + str(len(clean_array2)));    
    return clean_array2
    
        
def detrend_data(RR_array):
    new_coeffs=[]; 
    coeffs=pywt.wavedec(RR_array,'db3',mode='sp1',level=6);
    cA6,cD6,cD5,cD4,cD3,cD2,cD1 = coeffs;
    #print("cA is: " + str(cA));
    #print("cD is: " + str(cD));
    for i in range(len(cA6)):
        cA6[i]=0;    
    new_coeffs=cA6,cD6,cD5,cD4,cD3,cD2,cD1;
    RR_detrended=pywt.waverec(new_coeffs,'db3',mode='sp1');
    return RR_detrended