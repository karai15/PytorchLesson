import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""
https://colab.research.google.com/github/YutaroOgawa/pytorch_tutorials_jp/blob/main/notebook/0_Learn%20the%20Basics/0_4_buildmodel_tutorial_js.ipynb#scrollTo=dLbeJ_yEDQV0
"""



def main():
    # confirm GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # "cuda" or "cpu"
    print('Using {} device'.format(device))

    # create model
    model = NeuralNetwork().to(device)
    print(model)

    # input data to nn
    X = torch.rand(1, 28, 28, device=device)  # input data
    logits = model(X)  # ipnput data to nn (logits:output)
    # logits = model.forward(X)  # ipnput data to nn (logits:output)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    # confirm weight matrix in each layer
    print("Model structure: ", model, "\n\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

class NeuralNetwork(nn.Module):   # Inherit "nn.Module" class
    def __init__(self):
        super(NeuralNetwork, self).__init__()  # Inherit "nn.Module" class
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # input image (size 28*28)
            nn.ReLU(),
            nn.Linear(512, 512),  #
            nn.ReLU(),
            nn.Linear(512, 10),  # output (size 10)
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)  # matrix -> vector
        # ミニバッチの0次元目は、サンプル番号を示す次元で、この次元はnn.Flattenを通しても変化しません.1次元目以降がFlattenされます）。
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    main()