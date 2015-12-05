##this file downloads records names for given database name and reads rr intervals and saves mean and standard error or mean .
##Y axis of histogram is percentage of points falling in an interval


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
        if(unclean_array2[i] <0.3 or unclean_array2[i] >2):
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

def get_RR_interval(rec_name,annotation,start_time,end_time):
        #setup wfdb (change annotator here)
    rr_int=0;
    t=0;
    beats=0;
    sig_time=[];
    
    #change annotations
    nsamp, freq, annot, init_time = setupWfdb(rec_name,annotation);
    RR_sec_func=[];
    #num_sample=38400; # this translates to 230339 annot time which means 30 mins of data ## check this
    num_sample=end_time*60*freq
    ###### check if its the same record or new. if new , shift annot.time

#         annot.time
    
    #getann reads next annotation and returns 0 when successful
    # annot.time is time of the annotation, in samples from the beginning of the record.
    #when getting samples for cross-validation and testing, just adjust annot.time so it starts reading samples from there
    
    
    t = annot.time;
    ##comment june 16
    #print ("annot.time at the beginning is: " + str(annot.time))
    
    #annot_file=open('/home/ubuntu/Documents/eclispe_workspace/test_one/my_first_pyproj/research_March17/annot_type' +str(record)+'.txt','a');
    
    while wfdb.getann(0, annot) == 0 and annot.time<num_sample:
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
            RR_sec_func.append(rr_sec_rounded);
            # sampling intervals (e.g., if the original recording was sampled at 128 samples per second, then an 
            t = annot.time
            #print ("annot.time after rr interval is: " + str(t))
            
            #print ("-annot.time after rr interval is: " +  wfdb.timstr(-annot.time))
            #print ("annot.time in if is: " + str(annot.time))
            #prev_annot_time.append(annot.time)
    
    
    #print ("beats = "+ str(beats))
    return RR_sec_func



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


#for the given db_name
##download record names
##download annotator names

#for every record
##1. Open record 
##2. Extract RR interval
##3. Plot Histogram with constant bins
    ###3.1 y axis of Histogram shows frequency in percentage
##3. Extract Delta RR 



#use main from other file
#from main import svm_classifier function

#also import nonlinear measures from other file
#from


#######  Download rec names and annotator array for e database in db_name
db_name="afpdb";
rec_name_array=[];
annotator_array=[];

annotator_array=dload_annotator_names(db_name);
annotation=annotator_array[0];

rec_name_array=dload_rec_names(db_name)
print str(rec_name_array)

#############################################################################
mean_global_normal=[]
mean_global_patients=[]

for record in rec_name_array:
#for record in range(1):
    #variables and arrays 
    
    SDRR=[];
    SDRR_ms=[];
    ann_graph=[];
    RR_sec=[];
    #RR_sec=[ ];
    RR_sec_clean=[];
    RR_zero_m=[]; #RR_sec array with zero mean
    delta_RR_sec=[];
    rec_name=db_name+"/"+record;
    #rec_name= "afpdb/n22"
    

    start_time=0;
    end_time=30 #time in mins
    
    ##### Extract RR intervals here #######
    RR_sec=get_RR_interval(rec_name,annotation,start_time,end_time)
    
    x_val=[];
    y_val=[];
    
 
    ####DELETE THE FIRST RR_sec_orig value#####
    del RR_sec[0];
    
    #####Extract DELTA RR intervals here #########
    for i in range(len(RR_sec)):
        if i<len(RR_sec)-1:
            delta_RR_value=RR_sec[i+1]-RR_sec[i]
            delta_RR_sec.append(delta_RR_value)
    
    
    
    
    output_folder="/home/ubuntu/Documents/Thesis_work/RR_variability/09_oct_features/mean_standard_error_of_mean/"

#     fig=plt.figure();
#     weights = 100*np.ones_like(RR_sec)/len(RR_sec)
#     #plt.hist(RR_sec,weights=weights);
#     plt.hist(RR_sec,bins=100,weights=weights);
#     plt.xlabel("RR(sec)");
#     plt.ylabel("Percentage of total points");
#     plt.xlim(0,2);
#     plt.title("RR interval histogram for %s" % rec_name)



    ### plot RR intervals array ###
#     fig=plt.figure()
#     plt.scatter(range(len(RR_sec)),RR_sec,color='blue');
#     plt.xlabel("beats")
#     plt.ylabel(" RR interval");
#     plt.ylim(0,1.6)
#     plt.title("RR interval for %s" % rec_name);
#     fig.savefig(output_folder+"raw_rr_"+record+".png")
    #plt.show();


    
    #plt.show();
    
#     fig=plt.figure()
#     plt.plot(range(len(delta_RR_sec)),delta_RR_sec,color='red')
#     plt.xlabel("beats")
#     plt.ylabel("DELTA RR interval");
#     #plt.ylim(-0.1,0.1)
#     plt.title("DELTA RR interval for %s" % rec_name);
     
    ###### Calculating GOLBAL statistical features for RAW SDRR VALUES #################
    sdrr_raw=np.std(RR_sec);
    print (" sdrr of RR_sec (raw) in sec is: " +str(sdrr_raw));
    sdrr_raw_ms=sdrr_raw*1000;
    print (" sdrr of RR_sec (raw) in ms is: " +str(sdrr_raw_ms));
   
        

    ####calculating AVG/mean of RR intervals ###
    mean_global=np.mean(RR_sec)*1000;
    #print("mean_global is: " + str(mean_global))
    
    #### calculating skewness and kurtosis #####
    skewness_global=scipy.stats.skew(RR_sec)
    #print("skewness: "+ str(skewness_global))
    
    kurtosis_global=scipy.stats.kurtosis(RR_sec, fisher='false') 
    #print("kurtosis_globl is : " +str(kurtosis_global))   
    #print("answer for kurtosis test is: "+str(scipy.stats.kurtosistest(RR_sec))) 
    
    #################### WRTIE VALUES TO FILE  ########################################
    
    if "n" in record:
        #write to different file
        mean_global_normal.append(mean_global)
        f=open('/home/ubuntu/Documents/Thesis_work/RR_variability/09_oct_features/mean_standard_error_of_mean/normal_mean.txt','a');
        f.write(str(mean_global)+"\n")
        f.close
        
    elif "p" in record:
            #write to separate file
            mean_global_patients.append(mean_global)
            f=open('/home/ubuntu/Documents/Thesis_work/RR_variability/09_oct_features/mean_standard_error_of_mean/patient_mean.txt','a');
            f.write(str(mean_global)+"\n")
            f.close
    
    
    #################### USE filter FUNC TO CLEAN RR_SEC ARRAY  ########################################
     
    ##apply square filt 
    RR_sec_clean_one=square_filt(RR_sec);
     
    
    ##apply quotient filter 
    RR_sec_clean_two=quotient_filt(RR_sec_clean_one);

    
    ##apply quotient filter 2nd time
    RR_sec_clean=quotient_filt(RR_sec_clean_two);
    
    ##comment june 16
    #print(" np.stdRR_sec_clean is: "+ np.std(RR_sec_clean));
      
    
     
    ######### detrend  RR_sec_clean array with detrend_data func ###################################
    RR_sec_detrended=detrend_data(RR_sec_clean);
    
      
    #convert RR_sec_detrended in ms then later calc std
    RR_sec_detrended_ms=[x * 1000 for x in RR_sec_detrended];
    
    
    ############ Calculate SDRR of clean data #########################
    
    #RR_sec_detrended is in ms and std_dev will be in ms
    sdrr_curr=np.std(RR_sec_detrended_ms);

    #print("SDRR after filtering and detrending in ms for " + rec_name + "is: " + str(sdrr_curr));
    
    
#################### PLOT NORMAL GLOBAL MEAN GRAPHS  ##############################################
plt.figure()
plt.scatter(range(len(mean_global_normal)),mean_global_normal,color='blue');
plt.xlabel("normal records")
plt.ylabel(" Global mean values / ms");
#plt.ylim(0,1.6)
plt.title("means for normal records");
#fig.savefig(output_folder+"raw_rr_"+record+".png")

#################### PLOT AFIB GLOBAL MEAN GRAPHS  ##############################################

plt.figure()
plt.scatter(range(len(mean_global_patients)),mean_global_patients,color='red');
plt.xlabel("afib records")
plt.ylabel(" Global mean values / ms");
#plt.ylim(0,1.6)
plt.title("means for afib patients records");
#fig.savefig(output_folder+"raw_rr_"+record+".png")
plt.show();

#    
#    
# # #     #########PLOT unclean RR_intervals###################################
# #     plt.figure();
# #     plt.plot(range(len(RR_sec)),RR_sec);
# #     plt.ylim(-0.5,5);
# #     plt.xlabel("Samples");
# #     plt.ylabel(" RR interval (s)");
# #     plt.title("RR interval unclean plot for: %s" % rec_name);
# #     plt.draw();
# #     #plt.show();
#     
# #     difference= len(sig_time) -len(RR_sec_clean);
# #     for x in range(0,difference):
# #         del sig_time[x];
# #    
# #     print("length of sig_time is: " + str(len(sig_time)));
# #     print("length of RR_sec_clean is: "+ str(len(RR_sec_clean)));
# 
# 
# #     #########PLOT RR_intervals with clean and with TREND ###################################
# #     plt.figure();
# #     plt.plot(range(len(RR_sec_clean)),RR_sec_clean);
# #     #plt.plot(sig_time,RR_sec_clean);
# #     #plt.ylim(-1,2);
# #     #plt.xlim(0,5);
# #     plt.xlabel("Samples");
# #     #plt.xlabel("Time (hours since beginning of record)")
# #     plt.ylabel(" RR interval clean with trend (s)");
# #     plt.title("RR interval filtered (with trend) for: %s" % rec_name);
# #     plt.draw();
# #     #plt.show();
#     
#     
# #     difference2= len(sig_time) -len(RR_sec_detrended);
# #     for x in range(0,difference2):
# #         del sig_time[x];
# 
# 
# #     #########PLOT RR_intervals with clean and detrended ###################################
# #     plt.figure();
# #     plt.plot(range(len(RR_sec_detrended)),RR_sec_detrended);
# #     #plt.plot(sig_time,RR_sec_detrended)
# #     plt.ylim(-1,1);
# #     #plt.xlabel("Time (hours since beginning of record)")
# #     plt.xlabel("Samples");
# #     plt.ylabel(" RR interval clean (s)");
# #     plt.title("RR interval clean and zero mean plot for: %s" % rec_name);
# #     plt.draw();
# #     #plt.show();
#        
#     ############### SECOND ORDER DIFFERENCE CALCULATION ###############
#      
#     ## we start loop from 1  as we want to discard value at index 0 of RR interval
#     ## x_val, y_val have the coordinates of sodp plot
#      
#     j=0;
#     for n in range(1,len(RR_sec_detrended)):
#         if n<= (len(RR_sec_detrended)-2):
#             x_val.append(RR_sec_detrended[j+1]-RR_sec_detrended[j]);
#             y_val.append(RR_sec_detrended[j+2]-RR_sec_detrended[j+1]);
#             #print("n is: " +str(n))
#             #print("j is: " +str(j))
#             j=j+1;
#         else:
#             break;
#      
#     #print ("len of RR_sec_detrended is: " +str(len(RR_sec_detrended)));
#   
#     
# #     ###################Plot SODP graph for clean values#####################
# #     plt.figure();
# #     chf_plot=plt.scatter(x_val,y_val, color='blue' , marker='o');
# #     #axis labels and legends
# #     plt.xlabel("x[n+1]-x[n]");
# #     plt.ylabel(" x[n+2]-x[n+1] ");
# #     plt.title("SODP plot for clean: %s" % rec_name);
# #     plt.xlim(-0.5,0.5);
# #     plt.ylim(-0.5,0.5);
# #     plt.axhline(color = 'gray', zorder=-1)
# #     plt.axvline(color = 'gray', zorder=-1)
# #     rec_name1=rec_name;   
#     
#     ################# Calculate CTM measure #########################
#     count_ctm=0;
#     ctm_array=[];
#     
#     for j in range(0,len(radius_array)):
#         count_ctm=0;
#         for i in range(0,len(y_val)):
#             if ((((y_val[i])**2)+(x_val[i])**2)**0.5)<radius_array[j]:
#                 count_ctm=count_ctm+1;
#         
#         #after counting the number of points within radius_array[j] 
#         #ctm= fraction of points within the radius
#         ctm=float(count_ctm)/((len(y_val)));
#         ctm_array.append(ctm)
#     
#     #convert ctm_array to np.array  
#     np_ctm_array=np.array(ctm_array)
#     
#     #append row of np_ctm_array at row= record
#     ctm_all_records[record]=np_ctm_array;
#     
#     
#     ################# Calculate D(r) measure #########################
#     sum_distance=0;
#     distance_array=[];
#     
#     for j in range(0,len(radius_array)):
#         count_distance=0;
#         sum_distance=0;
#         for i in range(0,len(y_val)):
#             distance=((((y_val[i])**2)+(x_val[i])**2)**0.5);
#             if distance <radius_array[j]:
#                 sum_distance=sum_distance+distance;
#                 count_distance=count_distance+1;
#         
#         #after counting the number of points within radius_array[j] 
#         #distance= mean distance of points within the radius
#         mean_distance=float(sum_distance)/(count_distance);
#         distance_array.append(mean_distance)
#     
#     #convert distance_array to np.array  
#     np_dist_array=np.array(distance_array)
#     
#     #append row of np_distance_array at row= record
#     distance_all_records[record]=np_dist_array;
#     
#     
# #     #this is how you pick all rows 0th colom: ctm_all_records[:,0];
# #     
# #     plt.figure();
# #     plt.plot(radius_array,ctm_all_records[0,:]);
# #     #plt.draw();
# #     #plt.show();
#     
#     
# #     # ###### write to file ##############
# #     f=open('/home/ubuntu/Documents/eclispe_workspace/test_one/my_first_pyproj/ctm_nsrdb.txt','a');
# #     f.write("ctm for clean: %s" % rec_name + "\t");
# #     f.write(str(ctm_chf) + "\n");
# #     f.close();   
#     
# 
#     #uncomment this for for loop . this should happen at the end of each iteration within the for loop
#     SDRR_ms.append(sdrr_curr);
# 
# #       ###### write to file ##############
# #     f=open('/home/ubuntu/Documents/eclispe_workspace/test_one/my_first_pyproj/project_files/training_data/rr_test.txt','w');
# #     for i in range(0,len(SDRR_ms)):   
# #         ##for nsrdb write in a different file
# #         f.write(str(SDRR_ms[i])+ "\n");
# #     f.close();
# 
# #exit();
# #############   write feature arrays to a file #####################
# #f=open('/home/ubuntu/Documents/eclispe_workspace/test_one/my_first_pyproj/project_files/training_data/features_110515.txt','a');
# #ctm_all_records.tofile(f, sep=",", format="%f")
# #print "written to file"
# # for i in range(row_ctm_matrix):
# #     for j in range(col_ctm_matrix):
# #         f.write("ctm_all_record array:" + "\n");
# #         f.write(ctm_all_records[i,j])
# # f.close();
# 
# #     ###################Plot ctm curves vs r#####################
# plt.figure();
# line_c3, =plt.plot(radius_array,ctm_all_records[0,:],linestyle='-', marker='o',color='r',label='Class III patient')
# plt.plot(radius_array,ctm_all_records[1,:],linestyle='-', marker='o',color='r')
# plt.plot(radius_array,ctm_all_records[2,:],linestyle='-', marker='o',color='r')
# plt.plot(radius_array,ctm_all_records[3,:],linestyle='-', marker='o',color='r')
# plt.plot(radius_array,ctm_all_records[4,:],linestyle='-', marker='o',color='b')
# plt.plot(radius_array,ctm_all_records[5,:],linestyle='-', marker='o',color='b')
# plt.plot(radius_array,ctm_all_records[6,:],linestyle='-', marker='o',color='b')
# line_c2, =plt.plot(radius_array,ctm_all_records[7,:],linestyle='-', marker='o',color='b',label='Class II patient')
# plt.plot(radius_array,ctm_all_records[8,:],linestyle='-', marker='o',color='g')
# plt.plot(radius_array,ctm_all_records[9,:],linestyle='-', marker='o',color='g')
# plt.plot(radius_array,ctm_all_records[10,:],linestyle='-', marker='o',color='g')
# plt.plot(radius_array,ctm_all_records[11,:],linestyle='-', marker='o',color='g')
# plt.plot(radius_array,ctm_all_records[12,:],linestyle='-', marker='o',color='g')
# plt.plot(radius_array,ctm_all_records[13,:],linestyle='-', marker='o',color='g')
# line_c1, =plt.plot(radius_array,ctm_all_records[14,:],linestyle='-', marker='o',color='g',label='Class I patient')
# #axis labels and legends
# plt.xlabel("r");
# plt.ylabel(" CTM(r) ");
# plt.title("CTM vs. Radius for Class I,II and III Patients");
# plt.ylim(0,1.01);
# #plt.legend(handles=['Class III patient','Class II patient','Class I patient'])
# plt.legend([line_c3, line_c2,line_c1], ['Class III patient','Class II patient','Class I patient'],loc=4)
# 
# plt.show();
# #exit()
# 
# #     plt.xlim(-0.5,0.5);
# 
# #     plt.axhline(color = 'gray', zorder=-1)
# #     plt.axvline(color = 'gray', zorder=-1)
# #     rec_name1=rec_name;   
#     
# 
# #print "after insertion"
# #print ctm_all_records
# print class_label
# 
# p_val_ctm_array=[];
# p_val_dist_array=[];
# ################# separate CTM and D(r) values for class 0 arnd class 1 #########################
# for i in range(len(radius_array)):
#     ctm_class_0=[];
#     ctm_class_1=[];
#     dist_class_0=[];
#     dist_class_1=[];
#     
#     for j in range(ctm_all_records.shape[0]):
#            
#         if class_label[j] == '0':
#             ctm_class_0.append(ctm_all_records[j,i])
#             dist_class_0.append(distance_all_records[j,i])
#             
#         if class_label[j] == '1':
#             ctm_class_1.append(ctm_all_records[j,i])
#             dist_class_1.append(distance_all_records[j,i])
#                          
#     ################### CALCULATE ANOVA ###################
#     f_val, p_val_ctm = scipy.stats.f_oneway(ctm_class_0,ctm_class_1);
#     p_val_ctm_array.append(p_val_ctm);
#     
#     f_val_dist, p_val_dist = scipy.stats.f_oneway(dist_class_0, dist_class_1);
#     p_val_dist_array.append(p_val_dist); 
# 
# 
# # ################# separate CTM and D(r) values for each class #########################
# # for i in range(len(radius_array)):
# #     ctm_class_3=[];
# #     ctm_class_2=[];
# #     ctm_class_1=[];
# #     dist_class_3=[];
# #     dist_class_2=[];
# #     dist_class_1=[];
# #     
# #     for j in range(ctm_all_records.shape[0]):
# #         if class_label[j] == '3':
# #             ctm_class_3.append(ctm_all_records[j,i])
# #             dist_class_3.append(distance_all_records[j,i])
# #            
# #         
# #         if class_label[j] == '2':
# #             ctm_class_2.append(ctm_all_records[j,i])
# #             dist_class_2.append(distance_all_records[j,i])
# #             
# #         if class_label[j] == '1':
# #             ctm_class_1.append(ctm_all_records[j,i])
# #             dist_class_1.append(distance_all_records[j,i])
# #                          
# #     ################### CALCULATE ANOVA ###################
# #     f_val, p_val_ctm = scipy.stats.f_oneway(ctm_class_3, ctm_class_2, ctm_class_1);
# #     p_val_ctm_array.append(p_val_ctm);
# #     
# #     f_val_dist, p_val_dist = scipy.stats.f_oneway(dist_class_3, dist_class_2, dist_class_1);
# #     p_val_dist_array.append(p_val_dist); 
# print ctm_class_0
# #print ctm_class_2
# print ctm_class_1
# 
# ###### find minimum p value after ANOVA test for ctm ########
# min_p_val_ctm=np.amin(p_val_ctm_array);
# index_min_p=np.argmin(p_val_ctm_array);
# ##comment june 16
# print "best radius value is: " +str(radius_array[index_min_p]);
# print "p_val_ctm array is"
# print p_val_ctm_array;
# exit()
# ###### find minimum p value after ANOVA test for D(r) ########
# 
# min_p_val_dist=np.amin(p_val_dist_array);
# index_min_p_dist=np.argmin(p_val_dist_array);
# 
# # value of ctm and D(r)for least p value and corresponding r
# ctm_least_p=list(ctm_all_records[:,index_min_p]);
# dist_least_p=list(distance_all_records[:,index_min_p_dist]);
# ##comment june 16
# print "best radius value for dist arry is: " +str(radius_array[index_min_p_dist]);
# print "p_val_dist array is"
# print p_val_dist_array;
# 
# ############### write CTM array corresponding to least p value to file ################
# 
# ctm_feature_file=open('/home/ubuntu/Documents/eclispe_workspace/test_one/my_first_pyproj/project_files/training_data/features_160615/training/ctm_feature.txt','w');
# for ctm in ctm_least_p:
#     ctm_feature_file.write(str(ctm)+'\n');
# ctm_feature_file.close();
# 
# ############### write mean distance array corresponding to least p value to file ################
# 
# dist_feature_file=open('/home/ubuntu/Documents/eclispe_workspace/test_one/my_first_pyproj/project_files/training_data/features_160615/training/dist_feature.txt','w');
# for dist in dist_least_p:
#     dist_feature_file.write(str(dist)+'\n');
# dist_feature_file.close();
# 
# # ########### convert class labels to r,g,b and write to file ###################
# # class_label_color=[];
# # 
# # for i in range(len(class_label)):
# #     if class_label[i] == '1':
# #         class_label_color.append("r")
# #     if class_label[i] == '2':
# #         class_label_color.append("g")
# #     if class_label[i] == '3':
# #         class_label_color.append("b")
# # 
# # class_label_fo=open('/home/ubuntu/Documents/eclispe_workspace/test_one/my_first_pyproj/project_files/training_data/class_label_color.txt','w');
# # for color in class_label_color:
# #     class_label_fo.write(color + '\n');
# # class_label_fo.close();
# 
# 
# # ############### write SDRR_ms to file ################
# # sdrr_file=open('/home/ubuntu/Documents/eclispe_workspace/test_one/my_first_pyproj/project_files/training_data/features_190515/training/sdrr_feature.txt','w');
# # for sdrr in SDRR_ms:   
# #     sdrr_file.write(str(sdrr)+ "\n");
# # sdrr_file.close();
# # 
# # ############### write class label numbers to file ################
# # class_label_num_file=open('/home/ubuntu/Documents/eclispe_workspace/test_one/my_first_pyproj/project_files/training_data/class_label_num.txt','w');
# # for label in class_label:   
# #     class_label_num_file.write(str(label)+ "\n");
# # class_label_num_file.close();
# 
# 
# 
# ##other features
# exit();
# 
# 
# 
# ###### plot features plot to visualize data ####
# X=np.zeros(shape=(row_ctm_matrix, 2),dtype=np.float64); #initialze with zeros
# X[:,0]=ctm_least_p;
# X[:,1]=SDRR_ms;
# plt.scatter(X[:, 0], X[:, 1],s=40,c=class_label_color)
# plt.show();
# exit();
# 
# # ##uncomment this for record this should be outside for
# #plt.show();
# 
#  
# # ###### write to file ##############
# # f=open('/home/ubuntu/Documents/eclispe_workspace/test_one/my_first_pyproj/research_March23/sdrr_nsr2db_new_detrending.txt','a');
# # for i in range(0,len(SDRR_ms)):   
# #     ##for nsrdb write in a different file
# #     f.write(str(SDRR_ms[i])+ "\n");
# # f.close();
# # fo.close();
