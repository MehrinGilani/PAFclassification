## this file contains function definitions for following functions
#openWfdbSignal
#setupWFDB
#dload_rec_names
#dload_annot_names

# import matplotlib;
# import math;
# import matplotlib.pyplot as plt;
# import numpy as np;
# import wfdb,sys,re;
# import scipy;
# import scipy.sparse
# import scipy.stats
# import numpy.linalg as LA;
# import pywt;
# import os;
# 
# 
# from scipy import signal;
# from _wfdb import calopen, aduphys;
# from wfdb import WFDB_Siginfo
# from matplotlib.lines import lineStyles
# from _wfdb import strtim
# from matplotlib.pyplot import show
# from numpy import dtype, argmin

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



def dload_rec_names(database_name):
    rec_name_array=[];
    print("Wrote RECORDS.txt file in your current directory and saved rec_names in rec_name_array");
    cmd_records="wfdbcat "+database_name+"/RECORDS > RECORDS.txt"
    os.system(cmd_records);
    ###################### read rec names from file #####################
    rec_file=open("RECORDS.txt", 'r');
        
    for line in rec_file:
        temp=line.split();
        rec_name_array.append(temp[0]);

    
    return rec_name_array

def dload_annotator_names(database_name):
    annotator_array=[];
    cmd_annots="wfdbcat "+database_name+"/ANNOTATORS > ANNOTATORS.txt"
    os.system(cmd_annots);
    #read annot name from text file
    annot_file=open("ANNOTATORS.txt", 'r');
        
    for line in annot_file:
        temp=line.split();
        annotator_array.append(temp[0]);
        
    print("annotators for this database are: " + str(annotator_array) + " we are choosing " + str(annotator_array[0]))
    return annotator_array;
