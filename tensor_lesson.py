import torch
import numpy as np

"""
https://colab.research.google.com/github/YutaroOgawa/pytorch_tutorials_jp/blob/main/notebook/0_Learn%20the%20Basics/0_1_tensors_tutorial_js.ipynb#scrollTo=AlExSMGso65u
"""
# create tensor
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
ones_tensor = torch.ones((3, 3))

# numpy -> tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# tensor -> numpy
tensor = torch.ones(5)
numpy_t = tensor.numpy()  # translation

# product
tensor = torch.ones(4, 4)
y1 = tensor @ tensor  # matrix product
z1 = tensor * tensor  # all element product

test = 1