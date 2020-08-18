import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim = 1), targets[:,0,:,:])

class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor

class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)