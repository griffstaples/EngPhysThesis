#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Griffin Staples
Date Created: Thu Feb 21 2019
License:
The MIT License (MIT)
Copyright (c) 2019 Griffin Staples

"""

import numpy as np
import matplotlib.pylab as plt
import scipy.signal
import detect_peaks as dp
import pandas as pd

data = np.genfromtxt('./Data/CombinedData.txt')
data = data[:-2,:]
cols_to_use = [1, 2, 3, 4, 5, 6, 7, 8, 9]
Info = pd.read_excel('./Data/PPG-BP dataset.xlsx', skiprows=1, usecols=cols_to_use).values
skew = pd.read_excel('./Data/Table_1.xlsx',skiprows=0, usecols=[1,2,3,4]).values

patient = data[:, -1].copy()
data = data[:, :-1]
rows = len(data[:, 0])

def normalize1D(array):
    temp = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
    return np.nan_to_num(temp)+np.isnan(temp)*0.5


def normalizeRows(dumby, rows):
    array = dumby.copy()
    for i in range(rows):
        array[i, :] = normalize1D(array[i, :])
    return array


def normalizeCols(dumby, cols):
    array = dumby.copy()
    for i in range(cols):
        array[:, i] = normalize1D(array[:, i])
    return array


def getSystoleInfo(array, rows, bpm):
    # calculates for entire 2-D data array
    average_max = np.zeros(rows)  # average max of each signal
    pk_pk = np.zeros(rows)  # peak to peak interval
    holder = normalizeRows(array, rows)  # store normalized array
    ind = np.zeros((rows, 4))
    # define distance before another peak is looked for
    peak_distance = 60 * (1000) / (bpm * 2)

    for i in range(rows):
        temp = dp.detect_peaks(holder[i, :], mph=0.7, mpd=peak_distance[i])
        ind[i, :len(temp)] = temp
        ind = ind.astype(int)
        try:
            average_max[i] = np.sum(array[i, ind[i, np.nonzero(ind[i, :])]]) / np.count_nonzero(ind[i, :])
        except:
            average_max[i] = 0
    return average_max, ind


def getPatientData(Info, patient, rows, data_col):
    # create list of patient data that correspond to with patients array
    # array is as long as the dataset you give it (ie. rows)
    ordered_array = np.zeros(rows)
    for i in range(rows):
        for j in range(len(Info[:,0])):
            if patient[i] == Info[j, 0]:
                ordered_array[i] = Info[j, data_col]
                break
    return ordered_array


def getDiastoleInfo(deriv1, deriv2, sys_peak_index):
    # finds dichroic peak/notch
    ind = [elem for elem in sys_peak_index if elem != 0].copy()
    peak_index = np.zeros(len(ind))
    notch_index = np.zeros(len(ind))
    for i in range(len(ind)):
        try:
            peak_index[i] = dp.detect_peaks(deriv1[ind[i]:], mph=0.15)[0] +ind[i]
            notch_index[i] = dp.detect_peaks(deriv2[ind[i]:], mph=0.5)[0] + ind[i]
        except (ValueError, IndexError):
            peak_index = np.delete(peak_index, -1)
            notch_index = np.delete(notch_index, -1)
            ind = np.delete(ind, -1)

    peak_index = peak_index.astype(int)
    notch_index = notch_index.astype(int)
    ind = np.array(ind).astype(int)
    if not peak_index.any():
        peak_index = np.array([0])

    return peak_index, notch_index, ind

def getSpacings(dp_ind,sp_ind,dn_ind,normed_data,rows):
    #find spacing between first sp and first dp
    #find spacing between first sp and first dn
    #find width at half max
    if np.size(sp_ind)>=1:
        sp_ind=sp_ind[0]
    if np.size(dp_ind)>=1:
        dp_ind=dp_ind[0]
    if np.size(dn_ind)>=1:
        dn_ind = dn_ind[0]
    if not sp_ind:
        sp_ind = 0

    #print(sp_ind,' ',dp_ind,' ',dn_ind,' ')
    sp_dp_spacing = dp_ind-sp_ind
    sp_dn_spacing = dn_ind-sp_ind
    if not (sp_dp_spacing or sp_dn_spacing):
        sp_dp_spacing = 0
        sp_dn_spacing = 0
    #sp_dp_spacing[sp_dp_spacing==0] = np.nan
    #sp_dn_spacing[sp_dn_spacing==0] = np.nan
    #avg = np.nanmean(sp_dp_spacing)
    sp_ind = int(sp_ind)

    #avg = np.nanmean(sp_dn_spacing)
    low_ind = np.argmax(normed_data[(sp_ind-200):]>normed_data[sp_ind]/2)+sp_ind-200
    high_ind = np.argmax(normed_data[sp_ind:]<normed_data[sp_ind]/2)+sp_ind
    fwhm = high_ind-low_ind

    return sp_dp_spacing, sp_dn_spacing, fwhm

def plot_first_deriv(data_f):
    # find first derivative of filtered data and plot
    deriv1 = np.gradient(data_f, spacing, axis = 1)
    deriv1_f = scipy.signal.savgol_filter(deriv1, 99, 2)
    deriv1_f = scipy.signal.savgol_filter(deriv1_f, 99, 1)
    deriv1_f_n = normalizeRows(deriv1_f, rows)
    plt.plot(t, deriv1_f_n[testsignal, :])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude(t)' [1/s]")
    plt.show()
    return deriv1_f, deriv1_f_n

def plot_second_deriv(deriv1_f):

    # find second derivative of filtered data and plot
    deriv2 = np.gradient(deriv1_f, spacing, axis = 1)
    deriv2_f = scipy.signal.savgol_filter(deriv2, 99, 2)
    deriv2_f = scipy.signal.savgol_filter(deriv2_f, 59, 1)
    deriv2_f_n = normalizeRows(deriv2_f, rows)
    plt.plot(t, deriv2_f_n[testsignal, :])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude(t)\" [1/s]")
    plt.show()
    return deriv2_f, deriv2_f_n

SQI = np.zeros((len(patient)))

for p in range(2,len(patient)):
    for i in range(len(skew[:,0])):
        if(patient[p]==patient[p-2])and(patient[p]==skew[i,0]):
            SQI[p-2] = skew[i,1]
            SQI[p-1] = skew[i,2]
            SQI[p] = skew[i,3]
            break
        elif(patient[p] == patient[p-1])and(patient[p]==skew[i,0]):
            SQI[p-1] = skew[i,1]
            SQI[p] = skew[i,2]
            break
        elif(patient[p] ==skew[i,0]):
            SQI[p] = skew[i,1]
            break


testsignal = 0 #test signal to use
spacing = 1/1000 # spacing of data points for taking derivatives
t = np.linspace(0, 2.099, 2100) # time values corresponding to each data point

#filter and plot test signal
plt.figure(0)
data_f = np.array(scipy.signal.savgol_filter(data, 99, 2))
win = 99
data_m = np.convolve(data[testsignal,:],np.ones(win,)/win, mode = 'same')
data_f_n = normalizeRows(data_f, rows)
plt.plot(t,data[testsignal,:])
plt.plot(t, data_f[testsignal, :],'k')
plt.plot(t, data_m,'r')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(('Raw Data','S-G Filter','Moving Average'))
plt.xlim(0.35,0.8)
plt.ylim(1500,2700)
plt.show()


deriv1_f, deriv1_f_n = plot_first_deriv(data_f)
deriv2_f, deriv2_f_n = plot_second_deriv(deriv1_f)

#get info about systole signal
BPM = getPatientData(Info, patient, rows, 7)
avg_max, sys_peak_index = getSystoleInfo(data_f, rows, BPM)
aug_index = np.zeros(rows)

#get spacing between systolic and diastolic peaks and Full-Width Half-Max
sp_dp_spacing = np.zeros(rows)
sp_dn_spacing = np.zeros(rows)
fwhm = np.zeros(rows)

# calculate augmentation index
for i in range(rows):

    dp_ind, dn_ind, sp_ind = getDiastoleInfo(deriv1_f_n[i, :], deriv2_f_n[i, :], sys_peak_index[i, :])
    aug_index[i] = np.sum(data_f[i, dp_ind] / data_f[i, sp_ind]) / len(dp_ind)
    sp_dp_spacing[i],sp_dn_spacing[i],fwhm[i] = getSpacings(dp_ind,sp_ind,dn_ind,data_f_n[i,:],rows)

net_data = np.zeros((rows, 9))

# Transform Sex to 1 or 0 for neural net
for i in range(len(Info[:,0])):
    if Info[i, 1] == 'Female':
        Info[i, 1] = 0
    else:
        Info[i, 1] = 1

for i in range(9):
    net_data[:, i] = getPatientData(Info, patient, rows, i)

#reformat net data and answers
answers = net_data[:, 5:7].copy()
net_data = np.delete(net_data, (5, 6), 1)
avg_max = np.array(avg_max)

#add columns to net_data array
net_data = np.column_stack((net_data, avg_max)) #(6)
net_data = np.column_stack((net_data, aug_index)) #(7)
net_data = np.column_stack((net_data,sp_dp_spacing)) #(8)
net_data = np.column_stack((net_data,sp_dn_spacing)) #(9)
net_data = np.column_stack((net_data,fwhm))     #(10)
net_data[:, 1:] = normalizeCols(net_data[:, 1:], len(net_data[0, 1:]))
net_data = np.column_stack((net_data, answers))


# Save different net input and answers
np.random.shuffle(net_data)
np.savetxt('./Data/Net_Data.txt',net_data)



