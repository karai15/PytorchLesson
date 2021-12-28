import torch

"""
https://colab.research.google.com/github/YutaroOgawa/pytorch_tutorials_jp/blob/main/notebook/0_Learn%20the%20Basics/0_5_autogradqs_tutorial_jp.ipynb#scrollTo=hP0VaUxuYGRj
"""

# create network
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# calc gradient
loss.backward()
print(w.grad)  # calc gradient can be performed only in leaf node
print(b.grad)