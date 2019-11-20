#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:21:31 2019

@author: stevenalsheimer
"""

###create the rotation matrix and translation matrix to create R' here###
import numpy as np
import pandas as pd
import time
start_time = time.time()
r = ([[0,0,1],
      [1,0,0],
      [0,1,0]])
r = np.matrix(r)
print(r)
t = np.array([5,2,2])
#t = t.reshape(3,1)
print(t)

###math method for trandformation###

x = np.array([2,4,5])
x2 = np.dot(r,x)
print(x2)
x2 = x2 #+ t
x2 = np.array(x2)[0]
print(x2[0],x2[1],x2[2])

###loading the points###
DF = pd.read_csv('gh17.ptx', sep = ' ', header = None,names=["x", "y", "z", "i"])
row_num = DF.loc[0,'x']
col_num = DF.loc[1,'x']

num_points = row_num*col_num
DF_prime = pd.DataFrame(columns=["x", "y", "z", "i"])

#DF = DF[10:(int(num_points))]
DF = DF[10:18000]

#print(DF_shaved)

for i in range(10,len(DF)):
    x = DF.loc[i,'x']
    y = DF.loc[i,'y']
    z = DF.loc[i,'z']
    if x == 0 and y == 0 and z == 0:
        DF_prime.loc[i,'x'] = 0
        DF_prime.loc[i,'y'] = 0
        DF_prime.loc[i,'z'] = 0
        DF_prime.loc[i,'i'] = DF.loc[i,'i']
    else:
        point = np.array([x,y,z])
        point_prime = np.dot(r,point.T)###ROTATE THE POINT
        point_prime = np.array(point_prime)[0]
        point_prime = np.add(point_prime.T,t.T)###Translate the point
        print(point_prime)
        xp = point_prime[0]
        yp = point_prime[1]
        zp = point_prime[2]
        DF_prime.loc[i,'x'] = xp###convert point to new dataframe here
        DF_prime.loc[i,'y'] = yp
        DF_prime.loc[i,'z'] = zp
        DF_prime.loc[i,'i'] = DF.loc[i,'i']
print(DF_prime)
    
DF_prime.to_csv(r'gh17_R_prime.ptx', header=False, sep=' ', index=False)
DF.to_csv(r'gh17_R.ptx', header=False, sep=' ', index=False)
print("finished")
print("--- %s seconds ---" % (time.time() - start_time))




















