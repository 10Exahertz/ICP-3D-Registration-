#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 20:04:57 2019

@author: stevenalsheimer
"""

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
import csv
start_time = time.time()
print(start_time)
#r = ([[0.769006,0.300208,0.564362],
#      [-0.340899,0.939441,-0.035215],
#      [-0.540756,-0.165310,0.824776]])
#r = ([[ 9.99998671e-01, -1.20517537e-03,  1.09786929e-03],
#      [ 1.20577504e-03,  9.99999124e-01, -5.45711958e-04],
#      [-1.09721065e-03,  5.47035016e-04,  9.99999248e-01]])
r = ([[0.999953, 0.004953,  0.008324],
      [-0.003077, 0.977294, -0.211865],
      [-0.009184, 0.211829,  0.977264]])
r = np.matrix(r)
print(r)
t = np.array([-0.000003,0.001217,0.017408])
#t = t.reshape(3,1)
print(t)

###loading the points###
DF = pd.read_csv('gh17_23_1R.pts', sep = ' ', header = None,names=["x", "y", "z", "i"])
row_num = DF.loc[0,'x']
col_num = DF.loc[1,'x']

num_points = row_num*col_num
DF_prime = pd.DataFrame(columns=["x", "y", "z", "i"])

DF = DF[10:len(DF)]
#DF = DF[10:18000]

#print(DF_shaved)
with open('gh17_23_2R.pts','a',newline='')as csvFile:
    writer = csv.writer(csvFile,escapechar=' ',quoting = csv.QUOTE_NONE, delimiter = ' ')
    
    for i in range(10,len(DF)):
        x = DF.loc[i,'x']
        y = DF.loc[i,'y']
        z = DF.loc[i,'z']
        if x == 0 and y == 0 and z == 0:
            writer.writerow([0,0,0,DF.loc[i,'i']])
        else:
            point = np.array([x,y,z])
            point_prime = np.dot(r,point.T)###ROTATE THE POINT
            point_prime = np.array(point_prime)[0]
            #print(point_prime)
            point_prime = np.add(point_prime.T,t.T)###Translate the point
            #print(point_prime)
            xp = point_prime[0]
            yp = point_prime[1]
            zp = point_prime[2]
            writer.writerow([xp,yp,zp,DF.loc[i,'i']])
#print(DF_prime)
    
#DF_prime.to_csv(r'gh17_R.ptx', header=False, sep=' ', index=False)
#DF.to_csv(r'gh17_R.ptx', header=False, sep=' ', index=False)
print("finished")
print("--- %s seconds ---" % (time.time() - start_time))