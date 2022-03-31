# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:14:12 2022

@author: wuyic
"""

import torch
import matplot.pyplot as plt

dim = 8
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
A = torch.tensor(A,requires_grad=False).float()
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
b = torch.tensor(b,requires_grad=False).float()



x = torch.autograd.Variable(torch.zeros(dim, 1), requires_grad=True)
stop_loss = 1e-2
step_size = stop_loss / 10.0
lamda = 3000
print('Loss before: %s' % (torch.norm(torch.matmul(A, x) - b)))
Loss = []
itr = []
for i in range(5*10000):
    Δ = torch.matmul(A, x) - b
    L = torch.norm(Δ, p=2) #+ lamda*(torch.abs(torch.sum(x)))
    L.backward()
    x.data -= step_size * x.grad.data # step
    x.grad.data.zero_()
    if i % 1000 == 0: 
        itr.append(i)
        Loss.append(L.item())
        print('Loss is %s at iteration %i' % (L, i))
    if abs(L) < stop_loss:
        print('It took %s iterations to achieve %s loss.' % (i, step_size))
        break
print('Loss after: %s' % (torch.norm(torch.matmul(A, x) - b)))



plt.plot(itr,Loss)
plt.xlabel('Iteration',size=15)
plt.ylabel('Loss',size=15)