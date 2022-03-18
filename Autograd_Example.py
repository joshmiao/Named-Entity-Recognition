import torch
import time
import os
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

st = time.time()
a = torch.randn(10, 10, dtype=torch.float32)
for i in range(10000):
    a = torch.sqrt(a * a)
print(a - a)
print(time.time() - st, file=None)
if not os.path.exists('./theta_save/'):
    os.mkdir('./theta_save/')
torch.save(a,
           './theta_save/theta_save_tmp.pt')
