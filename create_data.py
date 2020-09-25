#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:35:50 2019
Created By: Griffin Staples
Student Number: 10168533

"""
import numpy as np
import matplotlib.pyplot as plt

num_files = 657     #should be 657
num_points = 2100
path = "./Data/"
count = 0
col_list = np.linspace(0,2099,2100).astype(int)

data = np.zeros((num_files,num_points+1))
#import all data from directory
for i in range(2,420): #should be 2 to 420
    for j in range(1,4):
        try:
            data[count,:num_points] = np.genfromtxt(path+str(i)+'_'+str(j)+'.txt')
            data[count,-1] = i
            count += 1
        except IOError:
            pass
        except:
            data[count,:num_points] = np.genfromtxt(path+str(i)+'_'+str(j)+'.txt', usecols = col_list)
            data[count,-1] = i
            count += 1


mat = np.matrix(data)
with open('CombinedDataTest.txt','wb') as f:
    for line in mat:
        np.savetxt(f,line)

