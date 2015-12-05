#this file extracrs short-term HRV features for normal  using get_hrv script
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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# from sklearn import svm, datasets
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import neighbors, datasets
# from sklearn import cross_validation
# from sklearn import metrics
# from sklearn.pipeline import make_pipeline

##################################   function definitions   #######################################

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


##########################################################################################

#variables
db_name="afpdb";
rec_name_array=[];
annotator_array=[];
start_time="00:00:00"
end_time="00:05:00"
input_file_path="/home/ubuntu/Documents/Thesis_work/RR_variability/"
#input_file_name= "RECORDS.txt"
output_folder_path="/home/ubuntu/Documents/Thesis_work/RR_variability/training_"+db_name+"/"


#function calls to download annotator and record names for the database in db_name
annotator_array=dload_annotator_names("afpdb");
annotation=annotator_array[0];

rec_name_array=dload_rec_names("afpdb")

###################### run get_hrv and grep features #####################

get_hrv_path="/home/ubuntu/mehrin/wfdb/wfdb-10.5.23/HRV"
feature_list="'SDNN|AVNN|rMSSD|pNN50|VLF PWR|LF PWR|HF PWR'"

all_file=output_folder_path+"all.txt"
time_interval=5;

#for i in range(len(rec_name_array)):
for i in range(0,1): #for testing
    rec_name=db_name+"/"+rec_name_array[i]
    ############# adding manual rec_name here#########
    rec_name="afpdb/p08"
    print "rec_name is: " + rec_name;
    
    ## opening records here ##
    #setup wfdb (change annotator here)
    nsamp, freq, annot, init_time = setupWfdb(rec_name, annotation);
    
    #execute command
    os.chdir(get_hrv_path) 
   
    #while time_interval < 61: #for other databases
    while time_interval < 31: #for mitdb
        
        cmd="./get_hrv -M "+rec_name +" "+ annotation+ " "+ start_time +" "+ end_time +" "+ "| " +"egrep " + feature_list +" >> " + all_file
        #print "command to be executed is:" + cmd
        os.system(cmd)
        time_interval=time_interval+5;
        new_end_time="00:"+str(time_interval)+":00"
        #print "new end time is : " + new_end_time
        start_time=end_time;
        end_time=new_end_time; 
        
    time_interval=5;
    start_time="00:00:00"
    end_time="00:05:00"


####### pick up features from all.txt and put in separate files ########
feature_array= ['SDNN','AVNN','pNN50','rMSSD' ,'VLF PWR','HF PWR','LF PWR']   

for feature in feature_array:
    feature_changed=feature
    feature_file=output_folder_path+(feature_changed.replace(" ","_").replace("/","_"))+ ".txt";
    cmd_grep_feature="grep "+"-w " +"\""+feature+"\""+" "+all_file +" "+ ">> " + feature_file
    print cmd_grep_feature
    os.system(cmd_grep_feature)
    
    
    
    
    
