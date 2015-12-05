
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

def check_rec_even_odd(rec_name):
    arr=list(rec_name);
    remain=float(arr[-1])%2
    if (remain == 0): # it is even
        even_or_odd="e"
    else:
        even_or_odd="o"
    return even_or_odd




def dload_rec_names(database_name):
    rec_name_array=[]
    print("Wrote RECORDS.txt file in your current directory and saved rec_names in rec_name_array");
    cmd_records="wfdbcat "+database_name+"/RECORDS > RECORDS.txt"
    #os.system(cmd_records);
    ###################### read rec names from file #####################
    rec_file=open("RECORDS.txt", 'r');
        
    for line in rec_file:
        temp=line.split();
        rec_name_array.append(temp[0]);

    
    return rec_name_array

def rmv_test_rec(initial_array):
    # this fucntion removes records with t from the initial_records_array
    without_t_array=[];
    for val in initial_array:
        if "t" not in val:
            without_t_array.append(val)
    print("done removing test records")
    return without_t_array
def rmv_odd_rec(initial_array):
    # this fucntion removes odd records from the initial_records_array
    without_odd_array=[];
    for val in initial_array:
        e_or_o=check_rec_even_odd(val)
        if "o" not in e_or_o:
            without_odd_array.append(val)
    print("done removing even records")
    return without_odd_array

def rmv_even_rec(initial_array):
    # this fucntion removes even records from the initial_records_array
    without_even_array=[];
    for val in initial_array:
        e_or_o=check_rec_even_odd(val)
        if "e" not in e_or_o:
            without_even_array.append(val)
    print("done removing odd records")
    return without_even_array

def rmv_continuation_rec(initial_array):
    # this fucntion removes records with c from the initial_records_array
    without_c_array=[];
    for val in initial_array:
        if "c" not in val:
            without_c_array.append(val)
    print("done removing contiuation records")

    return without_c_array
def keep_continuation_rec(initial_array):
# this fucntion removes records with c from the initial_records_array
    with_c_array=[];
    for val in initial_array:
        if "c" in val:
            with_c_array.append(val)
    print("just keeping contiuation records")

    return with_c_array



# def rmv_anomalous_rec(list_of_recs,input_array):
# # this fucntion removes given  records from the input array file
#     output_array=[];
#     for val in input_array:
#         print val
#         if (list_of_recs[0] and list_of_recs[1] and list_of_recs[2]) not in val:
#             print list_of_recs[0] 
#             print  list_of_recs[1]
#             print list_of_recs[2]
#             output_array.append(val)
#             print "appended in final array"
#     
#     print("done removing anomolous records")
#     return output_array

# 
# def populate_rec_name_arr(path_rec_file):
#     f=open(path_rec_file, 'r');
#     rec_name_array=[];
#     for line in f:
#         temp=line.split();
#         rec_name_array.append(temp[0]);
# 
#     return rec_name_array

def dload_annotator_names(database_name):
    annotator_array=[];
    cmd_annots="wfdbcat "+database_name+"/ANNOTATORS > ANNOTATORS.txt"
    os.system(cmd_annots);
    #read annot name from text file
    annot_file=open("ANNOTATORS.txt", 'r');
        
    for line in annot_file:
        temp=line.split();
        annotator_array.append(temp[0]);
    return annotator_array;


def openWfdbSignal(rec_name):
    #Find the number of signals in record
    nsig = wfdb.isigopen(rec_name, None, 0);
    if nsig<0:
        print "number of signals < 0, error opening signal record";
        exit();
    print "Number of signals: " + str(nsig) +" in record: "+ rec_name;
    return nsig;

def setupWfdb(rec_name, annotator ):
    nsig = openWfdbSignal(rec_name);
    
    #Allocate memory for sig info array
    #we can use siarray to access WFDB_Siginfo structure
    siarray = wfdb.WFDB_SiginfoArray(nsig);
    
    #Allocate memory for data
    sdata = wfdb.WFDB_SampleArray(nsig);
    
    #Open WFDB record
    wfdb.isigopen(rec_name, siarray.cast(), nsig);
    
    
    #read annotations from file
    #WFDB_Anninfor() contains name and attributes of annotator .atr etc
    a = wfdb.WFDB_Anninfo();
    
    #WFDB_Annotation describes the attributes of signals 
    #declare object in c : WFDB_Annotation annot; see below for declaring object in python
    annot = wfdb.WFDB_Annotation();
    
    #read name and status of annotation file
    #a.name="atr";
    #a.name="ecg";
    a.name=annotator;
    a.stat = wfdb.WFDB_READ;
    
    freq=wfdb.sampfreq(rec_name);
    nsamp=siarray[0].nsamp;
    print ("sampling frequency is: " + str(freq))
    init_time=wfdb.timstr(0);
    #print("strtim for starting value is: " + str(wfdb.strtim(init_time)));
    
    ##comment june 16
    ###### print signal specification #####
    record_info=wfdb.getinfo(rec_name)
    #print("getinfor is " + str(record_info));

    # print("total num of samples: " + str(nsamp));
    # print "Starting time of record is: "+ str(init_time);
    # print("sampling frequency is:"+ str(freq));      
    ########## READ ANNOTATION ##################
    if wfdb.annopen(rec_name, a, 1) < 0: 
        print("cannot open aanopen");
        exit();
    
    return (nsamp, freq, annot, init_time);