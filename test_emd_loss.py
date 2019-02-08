import torch

import time

from emd import EMDLoss

dist =  EMDLoss()

p1 = torch.rand(1,5,3).cuda().double()
p2 = torch.rand(1,10,3).cuda().double()
p1.requires_grad = True
p2.requires_grad = True

s = time.time()
cost = dist(p1, p2)
emd_time = time.time() - s

print('Time: ', emd_time)
print(cost)
loss = torch.sum(cost)
print(loss)
loss.backward()
print(p1.grad)
print(p2.grad)
