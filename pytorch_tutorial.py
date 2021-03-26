import torch

x = torch.empty(2, 2, 3)
print(x)

x = torch.rand(2, 3)
print(x)

x = torch.zeros(2, 3)
print(x)

x = torch.ones(2, 3)
print(x)

x = torch.ones(2, 3, dtype=torch.int)
print(x.dtype)

x = torch.ones(2, 3, dtype=torch.double)
print(x.dtype)

x = torch.ones(2, 3, dtype=torch.float16)
print(x.dtype)

x = torch.tensor([2.5, 0.1])
print(x.size)

x = torch.rand(2, 2)
y = torch.rand(2, 2)

z = x + y
print(z)

z = torch.add(x, y)
print(z)

#underscore means it does in place operations
y.add_(x)
print(y)

x = torch.rand(2,4)
print(x)
print(x[1,1].item())

y = x.view(8)
print(y)

z = y.view(-1, 2)
print(z.size())

import numpy as np

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))

a.add_(1)

print(a)

print(b)

a = np.ones(6)
print(a)

b = torch.from_numpy(a)
print(b)

a += 1
print(a)
print(b)
print(type(b))

if torch.cuda.is_available():
    print("I am in cuda")
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    z.to("cpu")
else:
    print("I am not in cuda")

x = torch.ones(5, requires_grad=True)
print(x)
