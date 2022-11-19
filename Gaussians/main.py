from random import shuffle
from models import NormalizingFlow
import torch 
torch.manual_seed(0)
import torch.nn as nn
from utils import wasserstein_loss
from train import *


# For GPU
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



# Generate data
d = 78
# base_dist = torch.distributions.MultivariateNormal(torch.zeros(d).to(device),torch.eye(d).to(device))
# sig = torch.eye(d)
# target_dist = torch.distributions.MultivariateNormal(torch.ones(d).to(device),sig.to(device))
base_dist = torch.distributions.MultivariateNormal(torch.zeros(d).to(device),torch.eye(d).to(device))
sig = torch.eye(d)*3 + torch.ones((d,d))
target_dist = torch.distributions.MultivariateNormal(torch.ones(d).to(device),sig.to(device))

samples = base_dist.sample((60000,))
data_loader = torch.utils.data.DataLoader(samples, batch_size = 60, shuffle = True)


flow_length = 100
netG = NormalizingFlow(d,flow_length).to(device)
lr = 0.0001
beta1 = 0.5
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
lam = 1
losses, kl_losses, w_losses = train_SP(netG,optimizerG,data_loader,base_dist,target_dist,lam,alpha_lam = 1.2,lr = 0.0001, num_epochs = 100,ngpu = 1, ot=True)
print(kl_losses[-1],w_losses[-1])

netG = NormalizingFlow(d,flow_length).to(device)
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
losses, kl_losses, w_losses = train_QP(netG,optimizerG,data_loader,base_dist,target_dist,lam,alpha_lam = 1.2,lr = 0.0001, num_epochs = 100,ngpu = 1, ot=True)
print(kl_losses[-1],w_losses[-1])

netG = NormalizingFlow(d,flow_length).to(device)
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
losses, kl_losses, w_losses = train_QP(netG,optimizerG,data_loader,base_dist,target_dist,lam,alpha_lam = 1.2,lr = 0.0001, num_epochs = 100,ngpu = 1, ot=True)
print(kl_losses[-1],w_losses[-1])

lr = 0.0001

netG = NormalizingFlow(d,flow_length).to(device)
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
rho = 0.001
losses, kl_losses, w_losses = train_AL(netG,optimizerG,data_loader,base_dist,target_dist,lam,rho,lr,num_epochs = 100)
print(kl_losses[-1],w_losses[-1])

netG1= NormalizingFlow(d,flow_length).to(device)
optimizerG1= torch.optim.Adam(netG1.parameters(), lr=lr, betas=(beta1, 0.999))
netG2 = NormalizingFlow(d,flow_length).to(device)
optimizerG2 = torch.optim.Adam(netG2.parameters(), lr=lr, betas=(beta1, 0.999))
lamsize = (1000,d)
rho = 10
losses, kl_losses, w_losses = train_ADMM(netG1, netG2, optimizerG1, optimizerG2,data_loader, base_dist,target_dist,lamsize,rho,lr,  num_epochs = 100)
print(kl_losses[-1],w_losses[-1])
