import torch
import time
import os
import math
# device = torch.device('cpu')
device = torch.device('cuda:0')
'''
# test grad 1
x = torch.tensor([2], dtype=torch.float, requires_grad=True, device=device)
y = 3 * x ** 2 + 2 * x
y.backward()
print(x.grad)

# test grad 2
x = torch.tensor([2, 3, 4], dtype=torch.float, requires_grad=True, device=device)
y = 3 * x ** 2 + 2 * x

print("y:", y)
torch.autograd.backward(y, retain_graph=True, grad_tensors=torch.tensor([1, 1, 1], dtype=torch.float32, device=device))
# y.backward(retain_graph=True, gradient=torch.tensor([1,1,1], dtype=torch.float32))
print("grad_tensors=torch.tensor([1,1,1]:", x.grad)  # tensor([14., 20., 26.])
torch.autograd.backward(y, retain_graph=True, grad_tensors=torch.tensor([3, 2, 1], dtype=torch.float32, device=device))
# y.backward(retain_graph=True, gradient=torch.tensor([3,2,1], dtype=torch.float32))
print("grad_tensors=torch.tensor([3,2,1]:", x.grad)  # tensor([42., 40., 26.])


a = torch.tensor([[1, 2, 3], [0, 0, 0]], dtype=torch.float32, device=device, requires_grad=True)
b = torch.tensor([4, 5, 6], dtype=torch.float32, device=device, requires_grad=True)

s = a[0] @ b
y = s
y += - s
print(y)
torch.autograd.backward(y, retain_graph=True, grad_tensors=torch.tensor(1, dtype=torch.float32, device=device))
print(a.grad)
print(torch.exp(torch.tensor(1)))
'''

a = torch.tensor(3, dtype=torch.float32)
ans = 0
st = time.time()
for i in range(1000000):
    ans += a * a
print(time.time() - st)
'''
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')
ans = 0
t1 = torch.tensor(2.2, dtype=torch.float32, device=gpu)
st = time.time()
for i in range(100000):
    ans += torch.sqrt(t1 * t1)
print(time.time() - st)
t2 = torch.tensor(2.2, dtype=torch.float64, device=gpu)
st = time.time()
for i in range(100000):
    ans += torch.sqrt(t2 * t2)
print(time.time() - st)
t3 = torch.tensor(2.2, dtype=torch.float32, device=cpu)
st = time.time()
for i in range(100000):
    ans += torch.sqrt(t3 * t3)
print(time.time() - st)
t4 = torch.tensor(2.2, dtype=torch.float64, device=cpu)
st = time.time()
for i in range(100000):
    ans += torch.sqrt(t4 * t4)
print(time.time() - st)
if t3 * t3 > 4:
    print(1)
'''