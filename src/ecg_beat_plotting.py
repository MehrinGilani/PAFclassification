import matplotlib;
import math;
import matplotlib.pyplot as plt;
import numpy as np;
import wfdb,sys,re;

from _wfdb import calopen, aduphys;
from wfdb import WFDB_Siginfo
from matplotlib.lines import lineStyles
from _wfdb import strtim
import data_cleaning as dc;
import os

def dload_rec_names(database_name):
    rec_name_array=[]
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
output_folder="/home/ubuntu/Documents/Thesis_work/results/thesis_images/chapter_4/"

#variables and arrays 
iteration=[];
sig_time=[];
count=0;
ann_graph=[];
split_time0=[];
annotator_array=[];

#Database and record name
db_name="afpdb";


annotation=dload_annotator_names(db_name)[0];
#rec_name=dload_rec_names(db_name)[0];
record="n08"
rec_name = "afpdb/"+record

#Find the number of signals in record
nsig = wfdb.isigopen(rec_name, None, 0);

if nsig<0:
    print "error opening signal record";
    exit();

print "Number of signals: " + str(nsig) +" in record: "+ rec_name;

#Allocate memory for sig info array
#we can use siarray to access WFDB_Siginfo structure
siarray = wfdb.WFDB_SiginfoArray(nsig);

#Allocate memory for data
sdata = wfdb.WFDB_SampleArray(nsig);

#Open WFDB record
wfdb.isigopen(rec_name, siarray.cast(), nsig);


#Move data from record to sdata
sig0 = [];
sig1 = [];

#physig0 is array with physical units
physig0=[];
physig1=[];

#read annotations from file
#WFDB_Anninfor() contains name and attributes of annotator .atr etc
a = wfdb.WFDB_Anninfo();

#WFDB_Annotation describes the attributes of signals 
#declare object in c : WFDB_Annotation annot; see below for declaring object in python
annot = wfdb.WFDB_Annotation();
#read name and status of annotation file

a.name=annotation;
print a.name
a.stat = wfdb.WFDB_READ;
freq=wfdb.sampfreq(rec_name);
nsamp=siarray[0].nsamp;
init_time=wfdb.timstr(0);
print type(init_time);
print("strtim for starting value is: " + str(wfdb.strtim(init_time)));

print("total num of samples: " + str(nsamp));
print "Starting time of record is: "+ str(init_time);
print("sampling frequency is:"+ str(freq));


def gettime(sample_num, freq, init_time):
    return float(sample_num)/float(freq)

#sample interval

#required length of signal in seconds
siglength_sec=1;
print type(freq);
loop_iteration=int(math.floor(siglength_sec*freq));

print("loop iteration = " +str(loop_iteration));


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
# for i in range(0,num_value):
#     if wfdb.getvec(sdata.cast()) < 0:
#         print "ERROR: getvec() < 0";
#         exit();
#     else:
#         #signal values in adu units:
#         sig0.append(sdata[0]);
#         sig1.append(sdata[1]);
#         sig_time.append(gettime(i, freq, init_time));
# 
# start_sec=4*60
# end_sec=5*60
# start_val=int(math.floor(start_sec*freq));
# end_val=int(math.floor(end_sec*freq));
# #loop_iteration=int(math.floor(siglength_sec*freq));
# for i in range(start_val,end_val):
# 
#         #print("time for sample " + str(i) + "is: " + str(sig_time[i]));
#         #convert adu units to physical units and save in physig0 and 1 (later generalise it for n number of signals)
#         physig0.append(aduphys(0,sig0[i]));
#         physig1.append(aduphys(1,sig1[i]));
#         wfdb.timstr(-annot.time),"(" + str(annot.time)+ ")",wfdb.annstr(annot.anntyp), annot.subtyp,annot.chan, annot.num
#         #append iteration number as value in 
#         iteration.append(i);
        
##########        READ ANNOTATION ##################
if wfdb.annopen(rec_name, a, 1) < 0: 
    print("cannot open aanopen");
    exit();
    
#getann reads next annotation and returns 0 when successful
while wfdb.getann(0,annot) ==0:
    if annot.time>num_value:
    #if annot.time>=start_val and annot.time<=end_val: 
        print("annot.time>number of samples extracted");
        break;
    #  annot.time is time of the annotation, in samples from the beginning of the record.
    print wfdb.timstr(-annot.time),"(" + str(annot.time)+ ")",wfdb.annstr(annot.anntyp), annot.subtyp,annot.chan, annot.num
   # print ("signal value at this annotation is : " + str(physig0[annot.time])+" "+ str(sig_time[annot.time]));
    
    
#else:
    #print("getann not working");
    #exit();

#write signals to file

# f=open('/home/ubuntu/Documents/eclispe_workspace/test_one/my_first_pyproj/sig_file.txt','a');
# f.write("sig_time values: \t signal 0 values: \t signal 1 values: \n");
# for i in range(0,len(physig0)):   
#     f.write(str(sig_time[i])+ "\t");
#     f.write(str(physig0[i]) + "\t");
#     f.write(str(physig1[i]) + "\n");
# # #f.write(str(ann_graph));
# f.close();

#starting time of record
print("starting time of record is: " + (wfdb.timstr(0L)));

#print array to check
#print("aduphys for sig0: " ,physig0);
#print("aduphys for sig1: " ,physig1);

##Plot graph
fig = plt.figure()


#write signal value to file
f=open("ecg_values.txt",'w')
for i in physig0:
    f.write(str(i))
    f.write("\n")

cmd="fft ecg_values.txt >  fft.txt"
os.system(cmd)

#read fft file
f=open("fft.txt",'r');
fft_values=[]
for i in f:
    fft_values.append(i)


#y values physical units
signal_num="P, QRS and T Waves"
#signal_num="Signal 1"
physig1=dc.detrend_data(physig1)
plt.plot(sig_time,physig1,linestyle="-",color='b');   
plt.xlabel("Time elapsed from start of record (sec)");
plt.ylabel(" Amplitude (mV) ");
plt.ylim(-1,1.5)
plt.title(signal_num); 
#plt.title(signal_num+" for " + rec_name); 
# plt.figure()
# plt.plot(sig_time,physig1,linestyle="-");
# 
# 
# #axis labels and legends
# plt.xlabel("Time elapsed from start of record (sec)");
# plt.ylabel(" Amplitude (mV) ");
# plt.title(" %s" % rec_name);

ax = fig.gca();
#ax.set_xticks(np.arange(0,(num_value/freq),0.2));
#ax.set_yticks(np.arange(sig_min,sig_max,0.5));

#keep grid
ax.grid(True);
#ax.set_xticklabels([])
plt.savefig(output_folder+"pqrst_ecg_"+record+".pdf",format='pdf')
    
# plt.figure()
# plt.plot(fft_values)
# plt.xlabel("? confirm if this is frequency");
# plt.ylabel(" fft ");
# plt.title(" %s" % rec_name);

plt.show();