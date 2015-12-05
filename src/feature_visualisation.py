# this file reads feature from file and plots it

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


from scipy import signal;
from _wfdb import calopen, aduphys;
from wfdb import WFDB_Siginfo
from matplotlib.lines import lineStyles
from _wfdb import strtim
from matplotlib.pyplot import show
from numpy import dtype, argmin, shape
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

feature_value=[];
#read feature
#db_name="nsrdb"
databases=["chfdb","afpdb"]
#databases=["afpdb"]
#feature_array= ['SDNN','AVNN','pNN50','rMSSD' ,'VLF PWR','HF PWR','LF PWR']   
feature_file="SDNN.txt"
color_code='b'
fig = plt.figure();
for i in databases:
    feature_file_path="/home/ubuntu/Documents/Thesis_work/RR_variability/training_"+i+"/"+feature_file;
    
    f=open(feature_file_path,'r')
    
    for line in f:
        temp=line.split()
        feature_value.append(temp[-1])
    
    #plot
    legend_label=i;
    #plt.plot(range(len(feature_value)),feature_value,linestyle="-",color=color_code, label=legend_label);
    plt.plot(feature_value,(np.zeros_like(feature_value)),'x',color=color_code, label=legend_label)
    plt.legend()
    color_code='r';
    #axis labels and legends
     
    feature_value=[];

#plt.xlabel("data points");
plt.xlabel(" %s" % feature_file);
plt.ylim(-0.001,0.001)
plt.title("  graph for " + feature_file + " in " + str(databases));
plt.show()
