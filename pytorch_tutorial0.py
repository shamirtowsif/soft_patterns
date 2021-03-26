import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * y * 2

print(z)

z = z.mean()

print(z)

z.backward()
print(x.grad)

x = torch.randn(3, requires_grad=True)
print(x)
x.requires_grad_(False)
print(x)

x = torch.randn(3, requires_grad=True)
print(x)
x = x.detach()
print(x)

x = torch.randn(3, requires_grad=True)
print(x)
with torch.no_grad():
    y = x + 2
    print(y)

weights = torch.ones(4, requires_grad=True)
