""" Model classes defined here! """

import torch
import torch.nn.functional as F
from math import floor
import pdb

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and 
        assign them as member variables.
        """
        super(FeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(28*28, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 8)

    def forward(self, x):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        # TODO: Implement this!
        x = F.relu(self.linear1(x))
        x = F.log_softmax(self.linear2(x))
        return x
        #raise NotImplementedError()

class SimpleConvNN(torch.nn.Module):
    def __init__(self, n1_chan, n1_kern, n2_kern):
        super(SimpleConvNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, n1_chan, kernel_size=n1_kern)
        self.conv2 = torch.nn.Conv2d(n1_chan, 8, kernel_size=n2_kern, stride=2)
        self.maxPool = torch.nn.MaxPool2d(kernel_size=n2_kern)

    def forward(self, x):
        # TODO: Implement this!
        x = x.view((-1, 1, 28, 28)) # Orig shape: [40, 784]
        x = self.conv1(x)
        # print(x.shape)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxPool(x)
        x = x.view((-1,8))
        return x

        # raise NotImplementedError()

def get_conv2d_or_max_pool2d_output_dimensions(h_in, w_out, kernel_size, stride):
   h_out = floor(((h_in - kernel_size - 1)/ stride) + 1)
   w_out = floor(((w_in - kernel_size - 1)/ stride) + 1)
   return (h_out, w_out)

class BestNN(torch.nn.Module):
    # TODO: You can change the parameters to the init method if you need to
    # take hyperparameters from the command line args!
    def __init__(self, n1_chan, n1_kern, n2_kern):
        super(BestNN, self).__init__()
        # TODO: Implement this!
        """
        self.model = torch.nn.Sequential(
                       torch.nn.Conv2d(1, n1_chan, kernel_size=n1_kern),
                       torch.nn.ReLU(),
                       torch.nn.BatchNorm2d(n1_chan),
                       torch.nn.Dropout2d(0.25),
                       torch.nn.Conv2d(n1_chan, n2_chan, kernel_size=n2_kern, stride=2),
                       torch.nn.ReLU(),
                       torch.nn.BatchNorm2d(n2_chan),
                       torch.nn.Dropout2d(0.25),
                       torch.nn.Conv2d(n2_chan, 8, kernel_size=n3_kern),
                       torch.nn.ReLU(),
                       torch.nn.BatchNorm2d(n1_chan),
                     )
        """

        self.conv1 = torch.nn.Sequential(
                        torch.nn.Conv2d(1, n1_chan, kernel_size=n1_kern),
                        torch.nn.ReLU(),
                        torch.nn.BatchNorm2d(n1_chan),
                        torch.nn.Dropout2d(0.15)
        )
        self.conv2 = torch.nn.Sequential(
                        torch.nn.Conv2d(n1_chan, 8, kernel_size=n2_kern, stride=2),
                        torch.nn.ReLU(),
                        torch.nn.BatchNorm2d(8)
        )
        self.linear = torch.nn.Sequential(
                        torch.nn.Linear(8,8)
                        # torch.nn.Dropout2d(0.2)
        )
        self.maxPool = torch.nn.Sequential(
                        torch.nn.MaxPool2d(kernel_size=n2_kern)
        )
        #raise NotImplementedError()

    def forward(self, x):
        x = x.view((-1, 1, 28, 28))
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.linear(x)
        x = self.maxPool(x)
        x = x.view((-1,8))
        return x
        #raise NotImplementedError()
