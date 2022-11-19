from random import shuffle
from models import NormalizingFlow
import torch.distributions as D
import torch 
torch.manual_seed(0)
import torch.nn as nn
from utils import wasserstein_loss
from train import *
import os 
import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='2DGaussian', help='data')
parser.add_argument('--length', type=int, default=10, help='length of nn')
parser.add_argument('--learning_rate', type=float, default=0.001, help='length of nn')
parser.add_argument('--method', type=str, default='ADMM', help='length of nn')
opt = parser.parse_args()
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def main():
    # For GPU
    d, base_dist, target_dist = distribution_generator(opt.data)
    # Generating data
    samples = base_dist.sample((60000,))
    data_loader = torch.utils.data.DataLoader(samples, batch_size = 2000, shuffle = True)
    flow_length = opt.length
    netG = NormalizingFlow(d,flow_length).to(device)
    lr = opt.learning_rate
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr,momentum=0.9)
    lam = 1
    if not os.path.exists('graph/%s_%s'%(opt.method,opt.data)):
        os.makedirs('graph/%s_%s'%(opt.method,opt.data))    
    folder = 'graph/%s_%s'%(opt.method,opt.data)

    if opt.method == 'SP':
        lam = 10

        netG = NormalizingFlow(d,flow_length).to(device)
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr, momentum = 0.9)
        losses, kl_losses, w_losses = train_SP(netG,optimizerG,data_loader,base_dist,target_dist,lam,alpha_lam=1,lr = lr, num_epochs = 100,ot=True)
        plotresult(netG,kl_losses,w_losses,base_dist,target_dist,lam,opt.method,folder)
        printresult(netG,base_dist,target_dist,opt.method,folder)
    if opt.method == 'QP':
        netG = NormalizingFlow(d,flow_length).to(device)
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr, momentum = 0.9)
        lam = 1.
        losses, kl_losses, w_losses = train_QP(netG,optimizerG,data_loader,base_dist,target_dist,lam,alpha_lam=1.2,lr = lr, num_epochs = 100,ot=True)
        plotresult(netG,kl_losses,w_losses,base_dist,target_dist,lam,opt.method,folder)
        printresult(netG,base_dist,target_dist,opt.method,folder)

    elif opt.method == 'SP_no':
    # netG = NormalizingFlow(d,flow_length).to(device)
    # optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr, momentum = 0.9)
    # losses, kl_losses, w_losses = train_QP(netG,optimizerG,data_loader,base_dist,target_dist,lam,lr = lr, num_epochs = 1000,ot=True)
    # plotresult(netG,kl_losses,w_losses,base_dist,target_dist,lam,'SP','graph_SP')
    # printresult(netG,base_dist,target_dist,'SP','graph_SP_%s'%opt.data)
    
    # if not os.path.exists('graph_SP_no_%s'%opt.data):
    #     os.makedirs('graph_SP_no_%s'%opt.data)

        netG = NormalizingFlow(d,flow_length).to(device)
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr, momentum = 0.9)
        losses, kl_losses, w_losses = train_SP(netG,optimizerG,data_loader,base_dist,target_dist,lam,lr = lr, num_epochs = 100,ot=False)
        plotresult(netG,kl_losses,w_losses,base_dist,target_dist,lam,'SP_no',folder)
        printresult(netG,base_dist,target_dist,'SP_no',folder)
    elif opt.method == 'AL':

    # if not os.path.exists('graph_AL_%s'%opt.data):
    #     os.makedirs('graph_AL_%s'%opt.data)

        netG = NormalizingFlow(d,flow_length).to(device)
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr, momentum = 0.9)
        rho = 0.001
        lam = 10
        losses, kl_losses, w_losses = train_AL(netG,optimizerG,data_loader,base_dist,target_dist,lam,rho,lr,num_epochs = 100)
        plotresult(netG,kl_losses,w_losses,base_dist,target_dist,lam,'AL',folder)
        printresult(netG,base_dist,target_dist,'AL',folder)

    # if not os.path.exists('graph_ADMM_%s'%opt.data):
    #     os.makedirs('graph_ADMM_%s'%opt.data)
    elif opt.method == 'ADMM':

        netG1= NormalizingFlow(d,flow_length).to(device)
        optimizerG1 = torch.optim.RMSprop(netG1.parameters(), lr=lr, momentum = 0.9)
        netG2 = NormalizingFlow(d,flow_length).to(device)
        optimizerG2= torch.optim.RMSprop(netG2.parameters(), lr=lr, momentum = 0.9)
        lamsize = (2000,d)
        rho = 20
        losses, kl_losses, w_losses, kl_1,w_1 = train_ADMM(netG1, netG2, optimizerG1, optimizerG2,data_loader, base_dist,target_dist,lamsize,rho,lr,  num_epochs = 300)
        # plotresult(netG2,lam,SP='ADMM')
        plotresult_ADMM(netG1,kl_losses,w_losses,kl_1,w_1,base_dist,target_dist,lam,'ADMM',folder)
        printresult(netG1,base_dist,target_dist,'ADMM',folder)



def distribution_generator(data):
    if data == '2DGaussian':
        d = 2
        base_dist = torch.distributions.MultivariateNormal(torch.zeros(2).to(device),torch.eye(2).to(device))
        sig = torch.eye(2)*3 + torch.ones((2,2))
        target_dist = torch.distributions.MultivariateNormal(torch.ones(2).to(device),sig.to(device))
        return d, base_dist, target_dist
    elif data == '2DMixture':
        d = 2
        base_dist = torch.distributions.MultivariateNormal(torch.zeros(2).to(device),torch.eye(2).to(device))
        n = 4
        x=np.array([[-1,-1],[-1,1],[1,1],[1,-1]]).astype('float')
        x=torch.from_numpy(x)

        mix = D.Categorical(torch.ones(n,).to(device))
        comp = D.Independent(D.Normal(
                        x.to(device), (torch.ones(n,2)*0.5).to(device)), 1)
        gmm = D.MixtureSameFamily(mix, comp)
        target_dist = gmm
        return d, base_dist, target_dist
    elif data == '2DMixture2Mixture':
        d = 2
        n = 4
        x=np.array([[-1,-1],[-1,1],[1,1],[1,-1]]).astype('float')
        x=torch.from_numpy(x)

        mix = D.Categorical(torch.ones(n,).to(device))
        comp = D.Independent(D.Normal(
                        x.to(device), (torch.ones(n,2)*0.5).to(device)), 1)
        gmm = D.MixtureSameFamily(mix, comp)
        x=np.array([[-2,0],[0,-2],[0,2],[2,0]]).astype('float')
        x=torch.from_numpy(x)

        mix = D.Categorical(torch.ones(n,).to(device))
        comp = D.Independent(D.Normal(
                        x.to(device), (torch.ones(n,2)).to(device)), 1)
        gmm = D.MixtureSameFamily(mix, comp)

        target_dist = gmm
        return d, base_dist, target_dist
    elif data == '784DGaussian':
        d = 784
        base_dist = torch.distributions.MultivariateNormal(torch.zeros(d).to(device),torch.eye(d).to(device))
        target_dist = torch.distributions.MultivariateNormal(2*torch.ones(d).to(device),torch.eye(d).to(device))
        return d, base_dist, target_dist
    elif data == '78DGaussian':
        d = 78
        base_dist = torch.distributions.MultivariateNormal(torch.zeros(d).to(device),torch.eye(d).to(device))
        sig = torch.eye(d)*3 + torch.ones((d,d))
        target_dist = torch.distributions.MultivariateNormal(torch.ones(d).to(device),sig.to(device))
        return d, base_dist, target_dist

def plotresult(netG, kl_losses, w_losses, base_dist,target_dist,lam, SP,folder):
    # plt.figure()
    # # plt.plot(kl_losses)
    # # # plt.savefig('graph/kl_loss%.1f.png'%lam)
    # # # plt.figure()
    # # plt.plot(w_losses)
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(kl_losses, '-*',label='kl loss')
    ax2.plot(w_losses, '-o',label='transport cost')

    ax1.set_xlabel('Iterations')
    # ax1.set_ylabel('loss', color='k')
    # ax2.set_ylabel('transport cost', color='k')
    # plt.text(400, 0.025, 'KL')
    # plt.text(7000, 0.042, 'transport cost')

    print(lam)

    plt.savefig('%s/%s_w_loss%.1f.png'%(folder,SP,lam))
    # plt.figure()
    # X=base_dist.sample((1600,)).to(device)
    # Y, _ =netG(X)
    # plt.plot(Y[:,0].detach().cpu().numpy(),Y[:,1].detach().cpu().numpy(),'.')
    # # sns.kdeplot(Y[:,0].detach().cpu().numpy(),Y[:,1].detach().cpu().numpy(),fill=True)
    # plt.savefig('%s/%s_generated%.1f.png'%(folder,SP,lam))

    # grid_size = 100
    # xx=torch.linspace(-2,2,grid_size)
    # yy=torch.linspace(-2,2,grid_size)
    # grid_x, grid_y = torch.meshgrid(xx, yy)#, indexing='ij')

    # XX = torch.stack((grid_x.flatten(),grid_y.flatten()),dim=1).to(device)
    # YY, logdet_YY = netG(XX)
    # logdetYY = (logdet_YY).reshape((100,100))
    # plt.figure()
    # plt.contourf(grid_x, grid_y,logdetYY.to('cpu').detach().numpy())
    # plt.tick_params(left = False, right = False , labelleft = False ,
    #                 labelbottom = False, bottom = False)

    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig('%s/%s_generated_Jacobian%.1f.png'%(folder,SP,lam))

def printresult(netG,base_dist,target_dist,method,folder):
    x = base_dist.sample((2000,))
    y,sum_log_det = netG(x)
    yy = target_dist.sample((2000,))
    l2loss = torch.nn.MSELoss()
    err= base_dist.log_prob(x).mean() + (- sum_log_det - (target_dist.log_prob(y))).mean()
    with open('%s/result.txt'%folder,'a') as f:
        f.writelines((method + ','+str(wasserstein_loss(x,y).item())+',',str(err.item())))
def plotresult_ADMM(netG, kl_losses, w_losses, kl1,w1, base_dist,target_dist,lam, SP,folder):
    plt.figure()
    # plt.plot(kl_losses)
    # # plt.savefig('graph/kl_loss%.1f.png'%lam)
    # # plt.figure()
    # plt.plot(w_losses)
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(kl_losses, '-.',label='kl loss')
    ax1.plot(kl1, '-*',label='kl loss')
    ax2.plot(w_losses, '-o',label='transport cost')
    ax2.plot(w1, '-v',label='transport cost')

    ax1.set_xlabel('Iterations')
    # ax1.set_ylabel('loss', color='k')
    # ax2.set_ylabel('transport cost', color='k')
    # plt.text(400, 0.025, 'KL')
    # plt.text(7000, 0.042, 'transport cost')

    # print(lam)

    plt.savefig('%s/%s_w_loss%.1f.png'%(folder,SP,lam))
    plt.figure()
    X=base_dist.sample((1600,)).to(device)
    Y, _ =netG(X)
    plt.plot(Y[:,0].detach().cpu().numpy(),Y[:,1].detach().cpu().numpy(),'.')
    # sns.kdeplot(Y[:,0].detach().cpu().numpy(),Y[:,1].detach().cpu().numpy(),fill=True)
    plt.savefig('%s/%s_generated%.1f.png'%(folder,SP,lam))

if __name__=='__main__':
    main()