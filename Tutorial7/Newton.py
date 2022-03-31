# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:30:09 2022

@author: wuyic
"""

import torch
import numpy as np
A =[[-1,1,0,0,0,0,0,0],
    [1,0,0,0,0,-1,0,0],
    [1,0,0,-1,0,0,0,0],
    [0,1,-1,0,0,0,0,0],
    [0,1,0,0,0,0,-1,0],
    [0,0,1,-1,0,0,0,0],
    [0,0,1,0,0,0,0,-1],
    [0,0,0,-1,1,0,0,0],
    [0,0,0,0,1,0,0,-1],
    [0,0,0,0,-1,1,0,0],
    [0,0,0,0,0,1,-1,0],
    [0,0,0,0,0,0,-1,1],
    [1,1,1,1,1,1,1,1]]
#    A = torch.tensor(A,requires_grad=False).float()
A = np.array(A)
b = [[10],
     [20],
     [4],
     [35],
     [34],
     [7],
     [12],
     [14],
     [7],
     [14],
     [10],
     [3],
     [0]]
#    b = torch.tensor(b,requires_grad=False).float()
b = np.array(b)
x = np.zeros((8,1))


B = np.matmul(A.T,A)
B_inv = np.linalg.inv(B)
#    delta = np.matmul(B_inv,np.matmul(A.T,np.matmul(A,x)-b))
x = np.matmul(B_inv,np.matmul(A.T,b))
np.round(x,2)
