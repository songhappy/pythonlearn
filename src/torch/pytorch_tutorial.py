# https://github.com/python-engineer/pytorchTutorial

## from youtube
# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
# default type for python is float64, but float32 for pytorch and tensorflow
import torch

x = torch.rand(2, 2)
y = torch.rand(2, 2)
y.add_(x)  # function with "_" means in place manipulation
z = x * y
print(x[:, 0])
print(x[0, 0].item())
x = torch.rand(4, 4)
# reshape
y = x.view(16)
a = torch.ones(5)
b = a.numpy()
a = torch.from_numpy(b)

x = torch.randn(3, requires_grad=True)
y = x + 2

z = y * y * 2
z = z.mean()  # scalar
z.backward()
print(x.grad)
x.requires_grad_(False)
x.detach()
with torch.no_grad():
    y = x + 2
    print(y)
