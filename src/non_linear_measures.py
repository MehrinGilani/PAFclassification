#This file reads qrs annotation of a wfdb record and calculates non-linear HRV measures: CTM(r) and D(r)
#CTM(r)=the fraction of total number of points in a second order difference plot that lie within a particular radius r
#D(r)=is the mean distance of points within a circular radius r
#usage: sodp.py <record name> <annotation> <radius array optional>

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
from scipy.misc import imread,imshow,imsave
from scipy.linalg import norm
from scipy import sum, average
import process_rr as pr;

def calc_sodp_values(rr_array):
    #take rr array and return x val and y val
     
    ############### SECOND ORDER DIFFERENCE CALCULATION ###############
 
    ## we start loop from 1  as we want to discard value at index 0 of RR interval
    ## x_val, y_val have the coordinates of sodp plot
    x_val=[];
    y_val=[];
    j=0;
    for n in range(1,len(rr_array)):
        if n<= (len(rr_array)-2):
            x_val.append(rr_array[j+1]-rr_array[j]);
            y_val.append(rr_array[j+2]-rr_array[j+1]);
            #print("n is: " +str(n))
            #print("j is: " +str(j))
            j=j+1;
        else:
            break;
     
    
  
    return x_val,y_val



def calc_sodp_measures(rec_name,rr_array,radius_array):
## take in rr_array , call calc_sodp_values function,     
    #return ctm_array, dist_array

    #### call cal_sodp_func #######
    x_val,y_val=calc_sodp_values(rr_array);
    ctm_feature_name=[]
    dist_feature_name=[]
    ################# Calculate CTM measure #########################
    count_ctm=0;
    ctm_array=[];
    
    for j in range(0,len(radius_array)):
        count_ctm=0;
        for i in range(0,len(y_val)):
            if ((((y_val[i])**2)+(x_val[i])**2)**0.5)<radius_array[j]:
                count_ctm=count_ctm+1;
        
        #after counting the number of points within radius_array[j] 
        #ctm= fraction of points within the radius
        ctm=float(count_ctm)/((len(y_val)));
        ctm_array.append(ctm)
    
    ################# Calculate D(r) measure #########################
    sum_distance=0;
    distance_array=[];
    
    for j in range(0,len(radius_array)):
        count_distance=0;
        sum_distance=0;
        for i in range(0,len(y_val)):
            distance=((((y_val[i])**2)+(x_val[i])**2)**0.5)
            if distance <radius_array[j]:
                sum_distance=sum_distance+distance;
                count_distance=count_distance+1;
        
        #after counting the number of points within radius_array[j] 
        #distance= mean distance of points within the radius
        if count_distance is 0:
            mean_distance=0;
            distance_array.append(mean_distance)
        else:
            mean_distance=float(sum_distance)/(count_distance);
            distance_array.append(mean_distance)

    
    for val in radius_array:
        ctm_feature_name.append("ctm_"+str(val))
        
    for val in radius_array:
        dist_feature_name.append("dist_"+str(val))
#     print("ctm array is: " + str(ctm_array));
#     print("distance array is: " + str(distance_array));
#     
#     print("----------------------------")
#     print("r" +  "\t" + "CTM(r)")
#     print("----------------------------")
#     for c in zip(radius_array, ctm_array):
#         print "%-7s %s" % c
#     print("----------------------------")
#     print("r" +  "\t" + "D(r)")
#     print("----------------------------")
#     for d in zip(radius_array, distance_array):
#         print "%-7s %s" % d
        
#     plt.figure();
#     plt.plot(radius_array,ctm_array);  
#     plt.xlabel("r");
#     plt.ylabel(" CTM(r)");
#     plt.title("CTM(r) vs. Radius plot for: %s" % rec_name);
#     plt.figure();
#     plt.plot(radius_array,distance_array);  
#     plt.xlabel("r");
#     plt.ylabel(" D(r)");
#     plt.title("D(r) vs. Radius for: %s" % rec_name);

    
    return x_val,y_val,ctm_array,ctm_feature_name,distance_array,dist_feature_name


def plot_ctm(rec_name_array,ctm_list_of_list,radius_array):
    plt.figure();
    plt.xlabel("r");
    plt.ylabel(" CTM(r)");
    plt.title("CTM(r) vs. Radius plot");
    for rec_name,ctm_list in zip(rec_name_array,ctm_list_of_list):
        if "n" in rec_name:
            normal, =plt.plot(radius_array,ctm_list,color='b')
        elif "p" in rec_name:
            patients, =plt.plot(radius_array,ctm_list,color='r')
    plt.legend([normal,patients],["normal", "patients"])
    return None

def plot_dist(rec_name_array,dist_list_of_list,radius_array):
    plt.figure();
    plt.xlabel("r");
    plt.ylabel(" Dist(r)");
    plt.title("Dist(r) vs. Radius plot");
    for rec_name,dist_list in zip(rec_name_array,dist_list_of_list):
        if "n" in rec_name:
            normal, =plt.plot(radius_array,dist_list,color='b')
        elif "p" in rec_name:
            patients, =plt.plot(radius_array,dist_list,color='r')
            
    plt.legend([normal,patients],["normal", "patients"])
    return None
def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr
def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

def calc_sodp_patient_diff(output_folder_path,file1,file2,file_name_to_save):
    #img1 = to_grayscale(imread(file1).astype(float))
    #img2 = to_grayscale(imread(file2).astype(float))
    img1 = to_grayscale(imread(output_folder_path+file1).astype(float))
    img2 = to_grayscale(imread(output_folder_path+file2).astype(float))
    
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
 
    diff = img2 - img1  # elementwise for scipy arrays
    #imshow(diff)
    print output_folder_path+file_name_to_save
    imsave((output_folder_path+file_name_to_save), diff)
    m_norm = sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)


def count_points_in_16quad(x_val,y_val,quad):
    #print ("x_val is: " + str(x_val))
    #print ("y_val is: " +str(y_val))
    if quad is 1:
        #if x_val>=0.5 and x_val<=1 and y_val>= 0.5 and y_val<=1:
        if x_val>=0.5 and y_val>= 0.5:
            sub_quad=11
        #elif x_val>=0 and x_val<=0.5 and y_val>= 0.5 and y_val<=1:
        elif x_val>=0 and x_val<=0.5 and y_val>= 0.5:
            sub_quad=12
        elif x_val>=0 and x_val<=0.5 and y_val>= 0 and y_val<=0.5:
            sub_quad=13
        #elif x_val>=0.5 and x_val<=1 and y_val>= 0 and y_val<=0.5:
        elif x_val>=0.5 and y_val>= 0 and y_val<=0.5:
            sub_quad=14     
    
    elif quad is 2:
        #if x_val>=-0.5 and x_val<=0 and y_val>= 0.5 and y_val<=1:
        if x_val>=-0.5 and x_val<=0 and y_val>= 0.5:
            sub_quad=21
         
        #elif x_val>=-1 and x_val<=-0.5 and y_val>= 0.5 and y_val<=1:
        elif x_val<=-0.5 and y_val>= 0.5:
            sub_quad=22 
         
        #elif x_val>=-1 and x_val<=-0.5 and y_val>= 0 and y_val<=0.5:
        elif x_val<=-0.5 and y_val>= 0 and y_val<=0.5:
            sub_quad=23 
        elif x_val>=-0.5 and x_val<=0 and y_val>= 0 and y_val<=0.5:
            sub_quad=24 
     
    elif quad is 3:
        if x_val>=-0.5 and x_val<=0 and y_val>= -0.5 and y_val<=0:
            sub_quad=31
        #elif x_val>=-1 and x_val<=-0.5 and y_val>= -0.5 and y_val<=0:
        elif x_val<=-0.5 and y_val>= -0.5 and y_val<=0:
            sub_quad=32
        #elif x_val>=-1 and x_val<=-0.5 and y_val>= -1 and y_val<=-0.5:
        elif x_val<=-0.5 and y_val<=-0.5:
            sub_quad=33
        #elif x_val>=-0.5 and x_val<=0 and y_val>= -1 and y_val<=-0.5:
        elif x_val>=-0.5 and x_val<=0 and y_val<=-0.5:
            sub_quad=34
     
    elif quad is 4:
        #if x_val>=0.5 and x_val<=1 and y_val>= -0.5 and y_val<=0:
        if x_val>=0.5 and y_val>= -0.5 and y_val<=0:
            sub_quad=41
        elif x_val>=0 and x_val<=0.5 and y_val>= -0.5 and y_val<=0:
            sub_quad=42
        #elif x_val>=0 and x_val<=0.5 and y_val>= -1 and y_val<=-0.5:
        elif x_val>=0 and x_val<=0.5 and y_val<=-0.5:
            sub_quad=43
        #elif x_val>=0.5 and x_val<=1 and y_val>= -1 and y_val<=-0.5:
        elif x_val>=0.5 and y_val<=-0.5:
            sub_quad=44
    return sub_quad



def calc_5min_sodp_measures(rec_name,annotation, total_min, radius_array):
    #this functions calculates features from 5 min of RR intervals for record and returns a list of all these features
    #for testing
    #start_time_arr=[0]
    #end_time_arr=[5]
    feature_list_5min=[]
    num_5min_interval=1 #keeps track of which chunck of 5 min interval it is
    feature_name=[]
    
    
    
    start_time_arr=range(0,total_min,5)
    end_time_arr=range(5,total_min+5,5)
    
    # everytime the loop runs it extracts feauters for those 5 mins
    
    for start_time, end_time in zip(start_time_arr,end_time_arr):
        error_rec_names=[]
        count_quad1=0;
        count_quad2=0;
        count_quad3=0;
        count_quad4=0;
       
        count_quad11=0;  
        count_quad12=0
        count_quad13=0
        count_quad14=0
        
        count_quad21=0;  
        count_quad22=0
        count_quad23=0
        count_quad24=0
        
        count_quad31=0;  
        count_quad32=0
        count_quad33=0
        count_quad34=0
        
        
        count_quad41=0;  
        count_quad42=0
        count_quad43=0
        count_quad44=0
        points_at_origin=0;
        quad=0;
        print("calculating 5 min sodp features")
        print ("start time is: " +str(start_time));
        print ("end time is: " +str(end_time));
        
        
        
        
        ##### Extract RR intervals here #######
        
        RR_sec_unclean=pr.get_RR_interval(rec_name,annotation,start_time,end_time)
        
        ####DELETE THE FIRST RR_sec_unclean value#####
        del RR_sec_unclean[0];
        
        RR_sec=RR_sec_unclean
        
        print("len of 5min RR_sec is: " + str(len(RR_sec)))
        num_RR_5min=len(RR_sec)
        feature_list_5min.append(num_RR_5min)
        feature_name.append("num_RR_5min_"+str(num_5min_interval))
        print("num_RR_5min_"+str(num_5min_interval))
        x_val_sodp,y_val_sodp,ctm_array,ctm_feature_name,distance_array,dist_feature_name=calc_sodp_measures(rec_name,RR_sec, radius_array);
        ## calculate the number of poitns in each quadrant
        num_points_on_graph=len(x_val_sodp)
        
        for x_val,y_val in zip(x_val_sodp,y_val_sodp):
            if x_val <-1 or x_val>1 or y_val<-1 or y_val>1:
                print("x_val is: " +str(x_val) + "y_val is: " + str(y_val))
                print("errorneous value in: " + str(rec_name) +"_"+str(num_5min_interval))
                error_rec_names.append(rec_name)
            
            elif x_val > 0 and y_val >0:
                quad=1;
                count_quad1=count_quad1+1;
                sub_quad=count_points_in_16quad(x_val,y_val,quad)
                if sub_quad is 11:
                    count_quad11=count_quad11+1;
                elif sub_quad is 12:
                    count_quad12=count_quad12+1;
                elif sub_quad is 13:
                    count_quad13=count_quad13+1;
                elif sub_quad is 14:
                    count_quad14=count_quad14+1;
            elif x_val <0  and y_val >0:
                quad=2;
                count_quad2=count_quad2+1;
                sub_quad=count_points_in_16quad(x_val,y_val,quad)
                if sub_quad is 21:
                    count_quad21=count_quad21+1
                elif sub_quad is 22:
                    count_quad22=count_quad22+1
                elif sub_quad is 23:
                    count_quad23=count_quad23+1
                elif sub_quad is 24:
                    count_quad24=count_quad24+1
                 
            elif x_val <0 and y_val <0:
                quad=3;
                count_quad3=count_quad3+1;
                sub_quad=count_points_in_16quad(x_val,y_val,quad)
                if sub_quad is 31:
                    count_quad31=count_quad31+1
                elif sub_quad is 32:
                    count_quad32=count_quad32+1
                elif sub_quad is 33:
                    count_quad33=count_quad33+1
                elif sub_quad is 34:
                    count_quad34=count_quad34+1
                 
                 
                 
            elif x_val>0 and y_val<0:
                quad=4;
                count_quad4=count_quad4+1;
                sub_quad=count_points_in_16quad(x_val,y_val,quad)
                if sub_quad is 41:
                    count_quad41=count_quad41+1
                elif sub_quad is 42:
                    count_quad42=count_quad42+1
                elif sub_quad is 43:
                    count_quad43=count_quad43+1
                elif sub_quad is 44:
                    count_quad44=count_quad44+1
            else:
                points_at_origin=points_at_origin+1 
                
        ratio_points_at_origin=float(points_at_origin)/num_points_on_graph
        #store the quad count values in features list
         
        
        
        
       
        feature_list_5min.append(points_at_origin)
        feature_name.append("points_at_origin_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(ratio_points_at_origin)
        feature_name.append("ratio_points_at_origin_5min_"+str(num_5min_interval))
        
        
        
        feature_list_5min.append(count_quad1)
        feature_name.append("num_points_in_quad1_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(count_quad2)
        feature_name.append("num_points_in_quad2_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(count_quad3)
        feature_name.append("num_points_in_quad3_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(count_quad4)
        feature_name.append("num_points_in_quad4_5min_"+str(num_5min_interval))
        
        #calculate quad_ratio 
        quad1_ratio=float(count_quad1)/num_points_on_graph;
        quad2_ratio=float(count_quad2)/num_points_on_graph;
        quad3_ratio=float(count_quad3)/num_points_on_graph;
        quad4_ratio=float(count_quad4)/num_points_on_graph;
        
        quad_11_ratio=float(count_quad11)/num_points_on_graph;
        quad_12_ratio=float(count_quad12)/num_points_on_graph;
        quad_13_ratio=float(count_quad13)/num_points_on_graph;
        quad_14_ratio=float(count_quad14)/num_points_on_graph;
        
        quad_21_ratio=float(count_quad21)/num_points_on_graph;
        quad_22_ratio=float(count_quad22)/num_points_on_graph;
        quad_23_ratio=float(count_quad23)/num_points_on_graph;
        quad_24_ratio=float(count_quad24)/num_points_on_graph;
        
        quad_31_ratio=float(count_quad31)/num_points_on_graph;
        quad_32_ratio=float(count_quad32)/num_points_on_graph;
        quad_33_ratio=float(count_quad33)/num_points_on_graph;
        quad_34_ratio=float(count_quad34)/num_points_on_graph;
        
        quad_41_ratio=float(count_quad41)/num_points_on_graph;
        quad_42_ratio=float(count_quad42)/num_points_on_graph;
        quad_43_ratio=float(count_quad43)/num_points_on_graph;
        quad_44_ratio=float(count_quad44)/num_points_on_graph;
        
        #store the quad ratio values in features list
        feature_list_5min.append(quad1_ratio)
        feature_name.append("quad1_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad2_ratio)
        feature_name.append("quad2_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad3_ratio)
        feature_name.append("quad3_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad4_ratio)
        feature_name.append("quad4_ratio_5min_"+str(num_5min_interval))
        
        
   
        feature_list_5min.append(quad_11_ratio)
        feature_name.append("quad11_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_12_ratio)
        feature_name.append("quad12_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad_13_ratio)
        feature_name.append("quad13_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_14_ratio)
        feature_name.append("quad14_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_21_ratio)
        feature_name.append("quad21_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_22_ratio)
        feature_name.append("quad22_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad_23_ratio)
        feature_name.append("quad23_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_24_ratio)
        feature_name.append("quad24_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad_31_ratio)
        feature_name.append("quad31_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_32_ratio)
        feature_name.append("quad32_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad_33_ratio)
        feature_name.append("quad33_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_34_ratio)
        feature_name.append("quad34_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad_41_ratio)
        feature_name.append("quad41_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_42_ratio)
        feature_name.append("quad42_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad_43_ratio)
        feature_name.append("quad43_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_44_ratio)
        feature_name.append("quad44_ratio_5min_"+str(num_5min_interval))
        
        
        num_5min_interval=num_5min_interval+1;
        print ("rec_names with errornos records are : " + str(error_rec_names))
        
    return feature_list_5min,feature_name



def calc_30min_sodp_measures(rec_name,annotation, start_time_given,end_time_given, x_val_sodp,y_val_sodp):
    
    feature_list_30min=[]
    num_5min_interval=1 #keeps track of which chunck of 5 min interval it is
    feature_name=[]
    
    
    
    start_time_arr=[start_time_given]
    end_time_arr=[end_time_given]
    
    
    for start_time, end_time in zip(start_time_arr,end_time_arr):
        print("start time for 30 quad features is: " + str(start_time))
        print("end time for 30 quad features is: " + str(end_time))
        error_rec_names=[]
        count_quad1=0;
        count_quad2=0;
        count_quad3=0;
        count_quad4=0;
       
        count_quad11=0;  
        count_quad12=0
        count_quad13=0
        count_quad14=0
        
        count_quad21=0;  
        count_quad22=0
        count_quad23=0
        count_quad24=0
        
        count_quad31=0;  
        count_quad32=0
        count_quad33=0
        count_quad34=0
        
        
        count_quad41=0;  
        count_quad42=0
        count_quad43=0
        count_quad44=0
        points_at_origin=0;
        quad=0;
        print("calculating 30 min sodp features")
        print ("start time is: " +str(start_time));
        print ("end time is: " +str(end_time));
        
        
        
        
        ##### Extract RR intervals here #######
        
        #RR_sec_unclean=pr.get_RR_interval(rec_name,annotation,start_time,end_time)
        
        ####DELETE THE FIRST RR_sec_unclean value#####
        #del RR_sec_unclean[0];
        
        #RR_sec=RR_sec_unclean
        
        #print("len of 5min RR_sec is: " + str(len(RR_sec)))
        #num_RR_5min=len(RR_sec)
        #feature_list_30min.append(num_RR_5min)
        #feature_name.append("num_RR_5min_"+str(num_5min_interval))
        print("num_RR_5min_"+str(num_5min_interval))
        #x_val_sodp,y_val_sodp,ctm_array,ctm_feature_name,distance_array,dist_feature_name=calc_sodp_measures(rec_name,RR_sec, radius_array);
        ## calculate the number of poitns in each quadrant
        num_points_on_graph=len(x_val_sodp)
        
        for x_val,y_val in zip(x_val_sodp,y_val_sodp):
            if x_val <-1 or x_val>1 or y_val<-1 or y_val>1:
                print("x_val is: " +str(x_val) + "y_val is: " + str(y_val))
                print("errorneous value in: " + str(rec_name) +"_"+str(num_5min_interval))
                error_rec_names.append(rec_name)
            
            elif x_val > 0 and y_val >0:
                quad=1;
                count_quad1=count_quad1+1;
                sub_quad=count_points_in_16quad(x_val,y_val,quad)
                if sub_quad is 11:
                    count_quad11=count_quad11+1;
                elif sub_quad is 12:
                    count_quad12=count_quad12+1;
                elif sub_quad is 13:
                    count_quad13=count_quad13+1;
                elif sub_quad is 14:
                    count_quad14=count_quad14+1;
            elif x_val <0  and y_val >0:
                quad=2;
                count_quad2=count_quad2+1;
                sub_quad=count_points_in_16quad(x_val,y_val,quad)
                if sub_quad is 21:
                    count_quad21=count_quad21+1
                elif sub_quad is 22:
                    count_quad22=count_quad22+1
                elif sub_quad is 23:
                    count_quad23=count_quad23+1
                elif sub_quad is 24:
                    count_quad24=count_quad24+1
                 
            elif x_val <0 and y_val <0:
                quad=3;
                count_quad3=count_quad3+1;
                sub_quad=count_points_in_16quad(x_val,y_val,quad)
                if sub_quad is 31:
                    count_quad31=count_quad31+1
                elif sub_quad is 32:
                    count_quad32=count_quad32+1
                elif sub_quad is 33:
                    count_quad33=count_quad33+1
                elif sub_quad is 34:
                    count_quad34=count_quad34+1
                 
                 
                 
            elif x_val>0 and y_val<0:
                quad=4;
                count_quad4=count_quad4+1;
                sub_quad=count_points_in_16quad(x_val,y_val,quad)
                if sub_quad is 41:
                    count_quad41=count_quad41+1
                elif sub_quad is 42:
                    count_quad42=count_quad42+1
                elif sub_quad is 43:
                    count_quad43=count_quad43+1
                elif sub_quad is 44:
                    count_quad44=count_quad44+1
            else:
                points_at_origin=points_at_origin+1 
                
        ratio_points_at_origin=float(points_at_origin)/num_points_on_graph
        #store the quad count values in features list
         
        
        
        
       
        feature_list_30min.append(points_at_origin)
        feature_name.append("points_at_origin_30min_"+str(num_5min_interval))
        
        feature_list_30min.append(ratio_points_at_origin)
        feature_name.append("ratio_points_at_origin_30min_"+str(num_5min_interval))
        
        
        
        feature_list_30min.append(count_quad1)
        feature_name.append("num_points_in_quad1_30min_"+str(num_5min_interval))
        
        feature_list_30min.append(count_quad2)
        feature_name.append("num_points_in_quad2_30min_"+str(num_5min_interval))
        
        feature_list_30min.append(count_quad3)
        feature_name.append("num_points_in_quad3_30min_"+str(num_5min_interval))
        
        feature_list_30min.append(count_quad4)
        feature_name.append("num_points_in_quad4_30min_"+str(num_5min_interval))
        
        #calculate quad_ratio 
        quad1_ratio=float(count_quad1)/num_points_on_graph;
        quad2_ratio=float(count_quad2)/num_points_on_graph;
        quad3_ratio=float(count_quad3)/num_points_on_graph;
        quad4_ratio=float(count_quad4)/num_points_on_graph;
        
        quad_11_ratio=float(count_quad11)/num_points_on_graph;
        quad_12_ratio=float(count_quad12)/num_points_on_graph;
        quad_13_ratio=float(count_quad13)/num_points_on_graph;
        quad_14_ratio=float(count_quad14)/num_points_on_graph;
        
        quad_21_ratio=float(count_quad21)/num_points_on_graph;
        quad_22_ratio=float(count_quad22)/num_points_on_graph;
        quad_23_ratio=float(count_quad23)/num_points_on_graph;
        quad_24_ratio=float(count_quad24)/num_points_on_graph;
        
        quad_31_ratio=float(count_quad31)/num_points_on_graph;
        quad_32_ratio=float(count_quad32)/num_points_on_graph;
        quad_33_ratio=float(count_quad33)/num_points_on_graph;
        quad_34_ratio=float(count_quad34)/num_points_on_graph;
        
        quad_41_ratio=float(count_quad41)/num_points_on_graph;
        quad_42_ratio=float(count_quad42)/num_points_on_graph;
        quad_43_ratio=float(count_quad43)/num_points_on_graph;
        quad_44_ratio=float(count_quad44)/num_points_on_graph;
        
        #store the quad ratio values in features list
        feature_list_30min.append(quad1_ratio)
        feature_name.append("quad1_ratio_30min_"+str(num_5min_interval))
        
        feature_list_30min.append(quad2_ratio)
        feature_name.append("quad2_ratio_30min_"+str(num_5min_interval))
    
        feature_list_30min.append(quad3_ratio)
        feature_name.append("quad3_ratio_30min_"+str(num_5min_interval))
        
        feature_list_30min.append(quad4_ratio)
        feature_name.append("quad4_ratio_30min_"+str(num_5min_interval))
        
        
   
        feature_list_30min.append(quad_11_ratio)
        feature_name.append("quad11_ratio_30min_"+str(num_5min_interval))
    
        feature_list_30min.append(quad_12_ratio)
        feature_name.append("quad12_ratio_30min_"+str(num_5min_interval))
        
        feature_list_30min.append(quad_13_ratio)
        feature_name.append("quad13_ratio_30min_"+str(num_5min_interval))
    
        feature_list_30min.append(quad_14_ratio)
        feature_name.append("quad14_ratio_30min_"+str(num_5min_interval))
    
        feature_list_30min.append(quad_21_ratio)
        feature_name.append("quad21_ratio_30min_"+str(num_5min_interval))
    
        feature_list_30min.append(quad_22_ratio)
        feature_name.append("quad22_ratio_30min_"+str(num_5min_interval))
        
        feature_list_30min.append(quad_23_ratio)
        feature_name.append("quad23_ratio_30min_"+str(num_5min_interval))
    
        feature_list_30min.append(quad_24_ratio)
        feature_name.append("quad24_ratio_30min_"+str(num_5min_interval))
        
        feature_list_30min.append(quad_31_ratio)
        feature_name.append("quad31_ratio_30min_"+str(num_5min_interval))
    
        feature_list_30min.append(quad_32_ratio)
        feature_name.append("quad32_ratio_30min_"+str(num_5min_interval))
        
        feature_list_30min.append(quad_33_ratio)
        feature_name.append("quad33_ratio_30min_"+str(num_5min_interval))
    
        feature_list_30min.append(quad_34_ratio)
        feature_name.append("quad34_ratio_30min_"+str(num_5min_interval))
        
        feature_list_30min.append(quad_41_ratio)
        feature_name.append("quad41_ratio_30min_"+str(num_5min_interval))
    
        feature_list_30min.append(quad_42_ratio)
        feature_name.append("quad42_ratio_30min_"+str(num_5min_interval))
        
        feature_list_30min.append(quad_43_ratio)
        feature_name.append("quad43_ratio_30min_"+str(num_5min_interval))
    
        feature_list_30min.append(quad_44_ratio)
        feature_name.append("quad44_ratio_30min_"+str(num_5min_interval))
        
        
        num_5min_interval=num_5min_interval+1;
        print ("rec_names with errornos records are : " + str(error_rec_names))
        
    return feature_list_30min,feature_name


def calc_std_5min_sodp_measures(rec_name,annotation, total_min, radius_array):
    #this functions calculates features from 5 min of RR intervals for record , and calculates std of each feature for all 5min intervlas in a 30min record.
    #it returns  all_features_6_intervals : each list corresponds to one feature and values within the list are denote the feature value per interval
    #it returns: std_dev_all_features :each value is the std dev of the feature in 6 intervals 
    #it also returns feature_name_overall : names of features whose std dev can be found in std_dev_all_Features
    
 
    feature_list_5min=[]
    num_5min_interval=1 #keeps track of which chunck of 5 min interval it is
    feature_name=[]
    
    
    
    start_time_arr=range(0,total_min,5)
    end_time_arr=range(5,total_min+5,5)
    
    # everytime the loop runs it extracts feauters for those 5 mins
    
    for start_time, end_time in zip(start_time_arr,end_time_arr):
        error_rec_names=[]
        count_quad1=0;
        count_quad2=0;
        count_quad3=0;
        count_quad4=0;
       
        count_quad11=0;  
        count_quad12=0
        count_quad13=0
        count_quad14=0
        
        count_quad21=0;  
        count_quad22=0
        count_quad23=0
        count_quad24=0
        
        count_quad31=0;  
        count_quad32=0
        count_quad33=0
        count_quad34=0
        
        
        count_quad41=0;  
        count_quad42=0
        count_quad43=0
        count_quad44=0
        points_at_origin=0;
        quad=0;
        print("calculating std of 5 min sodp features")
        print ("start time is: " +str(start_time));
        print ("end time is: " +str(end_time));
        
        
        
        
        ##### Extract RR intervals here #######
        
        RR_sec_unclean=pr.get_RR_interval(rec_name,annotation,start_time,end_time)
        
        ####DELETE THE FIRST RR_sec_unclean value#####
        del RR_sec_unclean[0];
        
        RR_sec=RR_sec_unclean
        
        print("len of 5min RR_sec is: " + str(len(RR_sec)))
        num_RR_5min=len(RR_sec)
        
        print("num_RR_5min_"+str(num_5min_interval))
        x_val_sodp,y_val_sodp,ctm_array,ctm_feature_name,distance_array,dist_feature_name=calc_sodp_measures(rec_name,RR_sec, radius_array);
        ## calculate the number of poitns in each quadrant
        num_points_on_graph=len(x_val_sodp)
        
        for x_val,y_val in zip(x_val_sodp,y_val_sodp):
            if x_val <-1 or x_val>1 or y_val<-1 or y_val>1:
                print("x_val is: " +str(x_val) + "y_val is: " + str(y_val))
                print("errorneous value in: " + str(rec_name) +"_"+str(num_5min_interval))
                error_rec_names.append(rec_name)
            
            elif x_val > 0 and y_val >0:
                quad=1;
                count_quad1=count_quad1+1;
                sub_quad=count_points_in_16quad(x_val,y_val,quad)
                if sub_quad is 11:
                    count_quad11=count_quad11+1;
                elif sub_quad is 12:
                    count_quad12=count_quad12+1;
                elif sub_quad is 13:
                    count_quad13=count_quad13+1;
                elif sub_quad is 14:
                    count_quad14=count_quad14+1;
            elif x_val <0  and y_val >0:
                quad=2;
                count_quad2=count_quad2+1;
                sub_quad=count_points_in_16quad(x_val,y_val,quad)
                if sub_quad is 21:
                    count_quad21=count_quad21+1
                elif sub_quad is 22:
                    count_quad22=count_quad22+1
                elif sub_quad is 23:
                    count_quad23=count_quad23+1
                elif sub_quad is 24:
                    count_quad24=count_quad24+1
                 
            elif x_val <0 and y_val <0:
                quad=3;
                count_quad3=count_quad3+1;
                sub_quad=count_points_in_16quad(x_val,y_val,quad)
                if sub_quad is 31:
                    count_quad31=count_quad31+1
                elif sub_quad is 32:
                    count_quad32=count_quad32+1
                elif sub_quad is 33:
                    count_quad33=count_quad33+1
                elif sub_quad is 34:
                    count_quad34=count_quad34+1
                 
                 
                 
            elif x_val>0 and y_val<0:
                quad=4;
                count_quad4=count_quad4+1;
                sub_quad=count_points_in_16quad(x_val,y_val,quad)
                if sub_quad is 41:
                    count_quad41=count_quad41+1
                elif sub_quad is 42:
                    count_quad42=count_quad42+1
                elif sub_quad is 43:
                    count_quad43=count_quad43+1
                elif sub_quad is 44:
                    count_quad44=count_quad44+1
            else:
                points_at_origin=points_at_origin+1 
                
        ratio_points_at_origin=float(points_at_origin)/num_points_on_graph
        #store the quad count values in features list
         
        
        
        
       
        
        
        feature_list_5min.append(ratio_points_at_origin)
        feature_name.append("ratio_points_at_origin_5min_"+str(num_5min_interval))
        
        

        
        #calculate quad_ratio 
        quad1_ratio=float(count_quad1)/num_points_on_graph;
        quad2_ratio=float(count_quad2)/num_points_on_graph;
        quad3_ratio=float(count_quad3)/num_points_on_graph;
        quad4_ratio=float(count_quad4)/num_points_on_graph;
        
        quad_11_ratio=float(count_quad11)/num_points_on_graph;
        quad_12_ratio=float(count_quad12)/num_points_on_graph;
        quad_13_ratio=float(count_quad13)/num_points_on_graph;
        quad_14_ratio=float(count_quad14)/num_points_on_graph;
        
        quad_21_ratio=float(count_quad21)/num_points_on_graph;
        quad_22_ratio=float(count_quad22)/num_points_on_graph;
        quad_23_ratio=float(count_quad23)/num_points_on_graph;
        quad_24_ratio=float(count_quad24)/num_points_on_graph;
        
        quad_31_ratio=float(count_quad31)/num_points_on_graph;
        quad_32_ratio=float(count_quad32)/num_points_on_graph;
        quad_33_ratio=float(count_quad33)/num_points_on_graph;
        quad_34_ratio=float(count_quad34)/num_points_on_graph;
        
        quad_41_ratio=float(count_quad41)/num_points_on_graph;
        quad_42_ratio=float(count_quad42)/num_points_on_graph;
        quad_43_ratio=float(count_quad43)/num_points_on_graph;
        quad_44_ratio=float(count_quad44)/num_points_on_graph;
        
        #store the quad ratio values in features list
        feature_list_5min.append(quad1_ratio)
        feature_name.append("quad1_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad2_ratio)
        feature_name.append("quad2_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad3_ratio)
        feature_name.append("quad3_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad4_ratio)
        feature_name.append("quad4_ratio_5min_"+str(num_5min_interval))
        
        
   
        feature_list_5min.append(quad_11_ratio)
        feature_name.append("quad11_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_12_ratio)
        feature_name.append("quad12_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad_13_ratio)
        feature_name.append("quad13_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_14_ratio)
        feature_name.append("quad14_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_21_ratio)
        feature_name.append("quad21_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_22_ratio)
        feature_name.append("quad22_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad_23_ratio)
        feature_name.append("quad23_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_24_ratio)
        feature_name.append("quad24_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad_31_ratio)
        feature_name.append("quad31_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_32_ratio)
        feature_name.append("quad32_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad_33_ratio)
        feature_name.append("quad33_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_34_ratio)
        feature_name.append("quad34_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad_41_ratio)
        feature_name.append("quad41_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_42_ratio)
        feature_name.append("quad42_ratio_5min_"+str(num_5min_interval))
        
        feature_list_5min.append(quad_43_ratio)
        feature_name.append("quad43_ratio_5min_"+str(num_5min_interval))
    
        feature_list_5min.append(quad_44_ratio)
        feature_name.append("quad44_ratio_5min_"+str(num_5min_interval))
        
        #print ("feature list for this 5 min interval is: " + str(feature_list_5min))
        #print("indexes are:                              " + str(range(len(feature_list_5min))))
        num_5min_interval=num_5min_interval+1;
        print ("rec_names with errornos records are : " + str(error_rec_names))
    
    dist_bw_feature=21
    start_ind=0
    end_ind=105
    feature_name_overall=[]
    #print (len(feature_list_5min))
    all_features_6_intervals=[]
    for i in range(0,21):
        name=str(feature_name[i])
        feature_name_overall.append(name[:-2])
        print("feature_name is: " + str(name[:-2]))
        feature_index_5min=range(start_ind,end_ind+21,dist_bw_feature)
        one_feature_all_intervals=[]
        for val in feature_index_5min:
            #print("val is : "+str(val))
            feature_val=feature_list_5min[val]
            #print("feature_val at index : " + str(val)+" is: "+str(feature_val))
            one_feature_all_intervals.append(feature_val)
             
        all_features_6_intervals.append(one_feature_all_intervals)    
        start_ind=start_ind+1
        end_ind=end_ind+1
        feature_index_5min=range(start_ind,end_ind+21,dist_bw_feature)
        #print ("new indexes for feature index is: " + str(feature_index_5min))
    
    std_dev_all_features=[]
    for list_feature_vals in all_features_6_intervals:
        std_dev_5min_feature=np.std(list_feature_vals)
        std_dev_all_features.append(std_dev_5min_feature)    
       
    #for name,val in zip(feature_name_overall,std_dev_all_features):
        #print("name of feature is : " + str(name))
        #print("val of feature is : " + str(val))
    return all_features_6_intervals,std_dev_all_features,feature_name_overall






































