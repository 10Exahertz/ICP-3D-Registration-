#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:31:19 2019

@author: stevenalsheimer
"""

import numpy as np
import pandas as pd
import time
start_time = time.time()

###Open the first file ###
DF = pd.read_csv('gh17_R.ptx', sep = ' ', header = None,names=["x", "y", "z", "i"])
row_num = DF.loc[0,'x']
col_num = DF.loc[1,'x']

num_points = row_num*col_num
DF_prime = pd.DataFrame(columns=["x", "y", "z", "i"])

#DF = DF[10:(int(num_points))]
DF = DF[10:18000]

###Open Second File####
DF2 = pd.read_csv('gh17_R_prime.ptx', sep = ' ', header = None,names=["x", "y", "z", "i"])
row_num2 = DF2.loc[0,'x']
col_num2 = DF2.loc[1,'x']

num_points2 = row_num2*col_num2
#DF2 = pd.DataFrame(columns=["x", "y", "z", "i"])

#DF2 = DF2[10:(int(num_points))]
DF2 = DF2[10:18000]


###Compare exactly corresponding points###

xp1 = DF.loc[202,'x']
yp1 = DF.loc[202,'y']
zp1 = DF.loc[202,'z']

xp2 = DF.loc[120,'x']
yp2 = DF.loc[120,'y']
zp2 = DF.loc[120,'z']

xp3 = DF.loc[15000,'x']
yp3 = DF.loc[15000,'y']
zp3 = DF.loc[15000,'z']


p1r1 = np.array([xp1,yp1,zp1])
p2r1 = np.array([xp2,yp2,zp2])
p3r1 = np.array([xp3,yp3,zp3])

x_vec1 = np.subtract(p2r1,p1r1)
x_vec1 = x_vec1/(np.sqrt(x_vec1[0]**2+x_vec1[1]**2+x_vec1[2]**2))
#y_vec = (p3-p1)-(np.dot((p3-p1),x_vec))*x_vec
y_vec_sub1 = np.subtract(p3r1,p1r1)
y_vec1 = y_vec_sub1 - (np.dot(y_vec_sub1,x_vec1))*x_vec1
y_vec1 = y_vec1/(np.sqrt(y_vec1[0]**2+y_vec1[1]**2+y_vec1[2]**2))
#y_vec = y_vec/abs(y_vec)
z_vec1 = np.cross(x_vec1,y_vec1)

y_vec1 = y_vec1.reshape(3,1)
x_vec1 = x_vec1.reshape(3,1)
z_vec1 = z_vec1.reshape(3,1)
R1 = np.concatenate((x_vec1,y_vec1,z_vec1),axis=1)
r1 = np.matrix(R1)
#print(x_vec1,y_vec1,z_vec1)
#print(R1)


###Points in second set###
xp1r2 = DF2.loc[202,'x']
yp1r2 = DF2.loc[202,'y']
zp1r2 = DF2.loc[202,'z']

xp2r2 = DF2.loc[120,'x']
yp2r2 = DF2.loc[120,'y']
zp2r2 = DF2.loc[120,'z']

xp3r2 = DF2.loc[15000,'x']
yp3r2 = DF2.loc[15000,'y']
zp3r2 = DF2.loc[15000,'z']


p1r2 = np.array([xp1r2,yp1r2,zp1r2])
p2r2 = np.array([xp2r2,yp2r2,zp2r2])
p3r2 = np.array([xp3r2,yp3r2,zp3r2])

x_vec2 = np.subtract(p2r2,p1r2)
x_vec2 = x_vec2/(np.sqrt(x_vec2[0]**2+x_vec2[1]**2+x_vec2[2]**2))
#y_vec = (p3-p1)-(np.dot((p3-p1),x_vec))*x_vec
y_vec_sub2 = np.subtract(p3r2,p1r2)
y_vec2 = y_vec_sub2 - (np.dot(y_vec_sub2,x_vec2))*x_vec2
y_vec2 = y_vec2/(np.sqrt(y_vec2[0]**2+y_vec2[1]**2+y_vec2[2]**2))
#y_vec = y_vec/abs(y_vec)
z_vec2 = np.cross(x_vec2,y_vec2)

y_vec2 = y_vec2.reshape(3,1)
x_vec2 = x_vec2.reshape(3,1)
z_vec2 = z_vec2.reshape(3,1)
R2 = np.concatenate((x_vec2,y_vec2,z_vec2),axis=1)
r2 = np.matrix(R2)
#print(x_vec2,y_vec2,z_vec2)
#print(R2)

###Find rotational matrix from R to R_prime###
R_rotate = np.dot(R2,R1.T)##how to make R1 ==R2
point_trans = np.dot(R_rotate,p1r1.T)
print(point_trans)
print("Rotation Matrix given:")
print(R_rotate)
print("         ")
print("Translation given")
t = np.subtract(p1r2,point_trans)
print(t)