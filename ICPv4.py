#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:11:32 2019

@author: stevenalsheimer
"""

import numpy as np
import pandas as pd
import time
#from scipy import linalg
import os
import copy
import csv
start_time = time.time()

###Parameters for ICP####
max_iterations = 12
RMSE_min = 0.00000001
dist_thresh = 0.3
npoints = 1000
filename_final = "gh17_23_ICPFINAL0.3_2.pts"

import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)

###Open the left file ###
DFL = pd.read_csv('gh17_23_3R.pts', sep = ' ', header = None,names=["x", "y", "z", "i"])
print(DFL.head(5))

DFL_prime = pd.DataFrame(columns=["x", "y", "z", "i"])

###Open right File####
DFR = pd.read_csv('gh16.pts', sep = ' ', header = None,names=["x", "y", "z", "i"])
DFR = DFR.reset_index(drop=True)   
t_final = [0,0,0]
convergenceE = pd.DataFrame(columns=['i','E'])
R_rotate_Final = np.identity(3)
diffE_p = 1
E_p = 2000000000
RMSE = 1000
f=0
while f < max_iterations and diffE_p > RMSE_min:
    f+=1
    DFL_shaved = pd.DataFrame(DFL)
    indexNames = DFL_shaved[DFL_shaved['x'] == 0 ].index
    DFL_shaved.drop(indexNames , inplace=True)
    DFL_shaved = DFL_shaved.reset_index(drop=False)
    left_training_data = DFL_shaved.sample(n=npoints)#currently set to 1000 points
    left_training_data = left_training_data.reset_index(drop=False)
    for i in range(len(left_training_data)):####Find Corresponding Points per iteration here
        x = left_training_data.loc[i,'x']
        y = left_training_data.loc[i,'y']
        z = left_training_data.loc[i,'z']
        L1 = [x,y,z]
        #dist = abs(x2-x)+abs(y2-y)+abs(z2-z)
        dist = np.linalg.norm(DFR[['x', 'y', 'z']].sub(np.array(L1)), axis=1)
        dist_min = np.argmin(dist)
        left_training_data.loc[i,'dist'] = min(dist)
        left_training_data.loc[i,'xr'] = DFR.loc[dist_min,'x']
        left_training_data.loc[i,'yr'] = DFR.loc[dist_min,'y']
        left_training_data.loc[i,'zr'] = DFR.loc[dist_min,'z']
    #left_training_data.nsmallest(400, 'dist')
    indexNames = left_training_data[ left_training_data['dist'] > dist_thresh ].index
 
    ####### Delete these row indexes from dataFrame
    left_training_data.drop(indexNames , inplace=True)
    left_training_data = left_training_data.reset_index(drop=True)
    print(len(left_training_data))
    
    #print(left_training_data)
    #print(left_training_data['dist'].mean())
    centroid_trained = pd.DataFrame(columns=['x', 'y', 'z','xr','yr','zr'])
    x_centL = left_training_data.sum(axis=0)[2]/len(left_training_data)###finding centroids to subtract
    y_centL = left_training_data.sum(axis=0)[3]/len(left_training_data)
    z_centL = left_training_data.sum(axis=0)[4]/len(left_training_data)
    CentroidL = np.array([x_centL,y_centL,z_centL])
    for i in range(len(left_training_data)):
        x = left_training_data.loc[i,'x'] - x_centL#subtract centroids
        y = left_training_data.loc[i,'y'] - y_centL
        z = left_training_data.loc[i,'z'] - z_centL
        centroid_trained.loc[i,'x'] = x#add them to a new dataframe
        centroid_trained.loc[i,'y'] = y
        centroid_trained.loc[i,'z'] = z
        
    x_centR = left_training_data.sum(axis=0)[7]/len(left_training_data)
    y_centR = left_training_data.sum(axis=0)[8]/len(left_training_data)
    z_centR = left_training_data.sum(axis=0)[9]/len(left_training_data)
    for i in range(len(left_training_data)):
        x = left_training_data.loc[i,'xr'] - x_centR
        y = left_training_data.loc[i,'yr'] - y_centR
        z = left_training_data.loc[i,'zr'] - z_centR
        centroid_trained.loc[i,'xr'] = x
        centroid_trained.loc[i,'yr'] = y
        centroid_trained.loc[i,'zr'] = z
    CentroidR = np.array([x_centR,y_centR,z_centR])
    print("centroidR:",CentroidR)
    print("centroidL:",CentroidL)
    
    rxlx =0#finding the corresponding points for covariance matrix
    rxly =0
    rxlz =0
    rylx =0
    ryly =0
    rylz =0
    rzlx =0
    rzly =0
    rzlz =0
    for i in range(len(left_training_data)):###making covariance matrix
        x = centroid_trained.loc[i,'x'] 
        y = centroid_trained.loc[i,'y']
        z = centroid_trained.loc[i,'z'] 
        point_left = np.array([x,y,z])
        xr = centroid_trained.loc[i,'xr']
        yr = centroid_trained.loc[i,'yr']
        zr = centroid_trained.loc[i,'zr'] 
        point_right = np.array([xr,yr,zr])
        rxlx += xr*x
        rxly += xr*y
        rxlz += xr*z
        rylx += yr*x
        ryly += yr*y
        rylz += yr*z
        rzlx += zr*x
        rzly += zr*y
        rzlz += zr*z
        #covariance matrix
    H = ([[rxlx,rxly,rxlz],
          [rylx,ryly,rylz],
          [rzlx,rzly,rzlz]])
    H = np.matrix(H)
    #print(H)
    U,s,Vt = np.linalg.svd(H)
    R_rotate = np.dot(Vt.T,U.T)
    R_rotate = np.dot(U,Vt)
#    R_rotate = ([[0,0,1],
#                 [1,0,0],
#                 [0,1,0]])
    R_rotate = np.matrix(R_rotate)
    R_rotate_Final = np.dot(R_rotate,R_rotate_Final)
    print("R:",R_rotate)
    #print("Det:",np.linalg.det(R_rotate))
    EE = np.dot(R_rotate,H)
    E_new = EE.trace()#trace of R*H
    E = E_new
    #print("error_tomax:",E)
    #print("diffE:", DiffE)
    
    point_trans = np.dot(R_rotate,CentroidL.T)

    t = np.subtract(CentroidR,point_trans)
    t = np.array(t)[0]
    t_final = np.add(t_final,t)
    print("t:",t)

    for i in range(len(left_training_data)):
        x = left_training_data.loc[i,'x']
        y = left_training_data.loc[i,'y']
        z = left_training_data.loc[i,'z']
        point = np.array([x,y,z])
        point_prime = np.dot(R_rotate,point.T)###ROTATE THE POINT
        point_prime = np.array(point_prime)[0]
        point_prime = np.add(point_prime.T,t.T)###Translate the point
        xp = point_prime[0]
        yp = point_prime[1]
        zp = point_prime[2]
        left_training_data.loc[i,'x'] = xp###convert point to new dataframe here
        left_training_data.loc[i,'y'] = yp
        left_training_data.loc[i,'z'] = zp
    #E_c = copy.deepcopy(E_p)
    E_p = 0
    for i in range(len(left_training_data)):
        x = left_training_data.loc[i,'x']
        y = left_training_data.loc[i,'y']
        z = left_training_data.loc[i,'z']
        xr = left_training_data.loc[i,'xr']
        yr = left_training_data.loc[i,'yr']
        zr = left_training_data.loc[i,'zr']
        d = np.sqrt((x-xr)**2+(y-yr)**2+(z-zr)**2)
        d = d**2
        E_p += d
    E_c = copy.deepcopy(RMSE)
    RMSE = np.sqrt(E_p/(len(left_training_data)))
    convergenceE.loc[f,'i'] = f
    convergenceE.loc[f,'E'] = RMSE
    convergenceE.plot(kind='line',x='i',y='E',color='blue')
    plt.show()  
    #print("E_p:",RMSE)
    diffE_p = abs(RMSE-E_c)
    print("diff RMSE_p:", diffE_p)
    temp = "temporaryfile.pts"
    gh = open(temp, "w+")
    gh.close()
    with open(temp,'a',newline='')as csvFile:
        writer = csv.writer(csvFile,escapechar=' ',quoting = csv.QUOTE_NONE, delimiter = ' ')
    
        for i in range(len(DFL)):
            x = DFL.loc[i,'x']
            y = DFL.loc[i,'y']
            z = DFL.loc[i,'z']
            if x == 0 and y == 0 and z == 0:
                writer.writerow([0,0,0,DFL.loc[i,'i']])
            else:
                point = np.array([x,y,z])
                point_prime = np.dot(R_rotate,point.T)###ROTATE THE POINT
                point_prime = np.array(point_prime)[0]
                #print(point_prime)
                point_prime = np.add(point_prime.T,t.T)###Translate the point
                #print(point_prime)
                xp = point_prime[0]
                yp = point_prime[1]
                zp = point_prime[2]
                writer.writerow([xp,yp,zp,DFL.loc[i,'i']])
    DFL = pd.read_csv(temp, sep = ' ', header = None,names=["x", "y", "z", "i"])
    print("--- %s seconds ---" % (time.time() - start_time))    
###PLOT CONVERGENCE of E####
convergenceE.plot(kind='line',x='i',y='E',color='blue')
plt.show()   
print(dist_thresh)
print(npoints)    
print(R_rotate_Final)
print(t_final)  
###Give these final R and t to all points


f = open(filename_final, "w+")
f.close()
with open(filename_final,'a',newline='')as csvFile:
    writer = csv.writer(csvFile,escapechar=' ',quoting = csv.QUOTE_NONE, delimiter = ' ')
    
    for i in range(len(DFL)):
        x = DFL.loc[i,'x']
        y = DFL.loc[i,'y']
        z = DFL.loc[i,'z']
        if x == 0 and y == 0 and z == 0:
            writer.writerow([0,0,0,DFL.loc[i,'i']])
        else:
            point = np.array([x,y,z])
            point_prime = np.dot(R_rotate_Final,point.T)###ROTATE THE POINT
            point_prime = np.array(point_prime)[0]
            #print(point_prime)
            point_prime = np.add(point_prime.T,t_final.T)###Translate the point
            #print(point_prime)
            xp = point_prime[0]
            yp = point_prime[1]
            zp = point_prime[2]
            writer.writerow([xp,yp,zp,DFL.loc[i,'i']])
#print(DF_prime)
    
#DF_prime.to_csv(r'gh17_Reduced_prime.ptx', header=False, sep=' ', index=False)
#DFR.to_csv(r'gh23_reduced_no_centroid.ptx', header=False, sep=' ', index=False)

    
    
    
    
    
    
    
print("finished")
print("--- %s seconds ---" % (time.time() - start_time))  
os.system('say "done"')  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
