import torch

# device = torch.device('cpu')
device = torch.device('cuda:0')

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


t1 = torch.tensor([[1, 2]])
t2 = torch.tensor([2])
print(t2.item())