import torch 
torch.manual_seed(0)
import torch.nn as nn
def wasserstein_loss(input,output):
    l2loss = torch.nn.MSELoss()
    return l2loss(input,output)
