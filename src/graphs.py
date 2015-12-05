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
from itertools import izip_longest

def plotHistPercent(rec_name, RR_sec, n_bins, xlabel, ylabel, title, xlim_lo=0, xlim_hi=0, ylim_lo=0, ylim_hi=120):
    weights = 100*np.ones_like(RR_sec)/len(RR_sec)
    fig,plot = plotHist(rec_name, RR_sec, n_bins, xlabel, ylabel, title,xlim_lo,xlim_hi,ylim_lo,ylim_hi,weights);
    return fig, plot;

def plotHist(rec_name, RR_sec, n_bins,xlabel, ylabel, title, xlim_lo=0, xlim_hi=0, ylim_lo=0, ylim_hi=100,  weights=0):
    fig=plt.figure();
    #weights = 100*np.ones_like(RR_sec)/len(RR_sec)
    #plt.hist(RR_sec,weights=weights);
    if weights is not 0:
        plot = plt.hist(RR_sec,bins=n_bins,weights=weights);
    else:
        plot = plt.hist(RR_sec,bins=n_bins);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    #plt.xlabel("RR(sec)");
    #plt.ylabel("Percentage of total points");
    if xlim_hi is not 0:
        plt.xlim(xlim_lo,xlim_hi);
    #plt.title("RR interval histogram for %s" % rec_name)
    plt.title(title+" for %s" % rec_name);
    return fig, plot;
    
def plotScatter(rec_name,x_array,y_array, xlabel, ylabel, title, color_code,xlim_lo=0, xlim_hi=0, ylim_lo=0, ylim_hi=0,axline=0):
    ### plot RR intervals array ###
    fig=plt.figure()
    plot = plt.scatter(x_array,y_array,color=color_code);
    #plt.legend()
    
    
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    #plt.xlabel("beats")
    #plt.ylabel(" RR interval");
    if ylim_hi is not 0:
        plt.ylim(ylim_lo,ylim_hi);
    if xlim_hi is not 0:
        plt.xlim(xlim_lo,xlim_hi);

    plt.title(title+" for %s" % rec_name);
    if axline is not 0:
        plt.axhline(color = 'gray', zorder=-1)
        plt.axvline(color = 'gray', zorder=-1)
    return fig, plot;


def plot_simple(rec_name,x_array,y_array, xlabel, ylabel, title, color_code,xlim_lo=0, xlim_hi=0, ylim_lo=0, ylim_hi=0):
    fig=plt.figure()
    plot = plt.plot(x_array,y_array,color=color_code);
    #plt.legend()
    
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    #plt.xlabel("beats")
    #plt.ylabel(" RR interval");
    if ylim_hi is not 0:
        plt.ylim(ylim_lo,ylim_hi);
    if xlim_hi is not 0:
        plt.xlim(xlim_lo,xlim_hi);

    plt.title(title+" for %s" % rec_name);
    return fig, plot;

