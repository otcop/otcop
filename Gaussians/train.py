import torch 
torch.manual_seed(0)
import torch.nn as nn
from utils import wasserstein_loss


def train_SP(netG,optimizerG,dataloader,base_dist,target_dist,lam,alpha_lam = 1,lr = 0.0001, num_epochs = 100,ngpu = 1, ot=True):
  device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

  losses = []
  kl_losses = []
  w_losses = []
  for epoch in range(num_epochs):
    kl_loss = 0.
    w_loss = 0.
    loss_1 = 0.
    for i, noise in enumerate(dataloader):
        netG.zero_grad()  
        noise = noise.to(device) 
        z_k, sum_log_det = netG(noise)
        log_p_x = target_dist.log_prob(z_k)
        if ot:
            loss = lam*(base_dist.log_prob(noise).mean() + (- sum_log_det - (log_p_x)).mean()) + wasserstein_loss(noise,z_k)
        else:
            loss =  (base_dist.log_prob(noise).mean() + (- sum_log_det - (log_p_x)).mean())
        lam = lam 
        loss.backward()
        
        kl_loss += base_dist.log_prob(noise).mean() + (- sum_log_det - (log_p_x)).mean()
        w_loss += wasserstein_loss(noise,z_k)
        loss_1 += loss

        optimizerG.step()

    kl_losses.append(kl_loss.item()/i)
    w_losses.append(w_loss.item()/i)
    losses.append(loss_1.item()/i)
       
    if epoch%1==0:
        lam  = lam * alpha_lam
        print('epoch %d, loss: %f, kl_loss: %f, w_loss: %f'%(epoch,loss.item(),kl_loss/i,w_loss/i))
  return losses, kl_losses, w_losses



def train_QP(netG,optimizerG,dataloader,base_dist,target_dist,lam,alpha_lam = 1,lr = 0.0001, num_epochs = 100,ngpu = 1, ot=True):
  device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

  losses = []
  kl_losses = []
  w_losses = []
  for epoch in range(num_epochs):
    kl_loss = 0.
    w_loss = 0.
    loss_1 = 0.
    for i, noise in enumerate(dataloader):
        netG.zero_grad()  
        noise = noise.to(device) 
        z_k, sum_log_det = netG(noise)
        log_p_x = target_dist.log_prob(z_k)
        if ot:
            loss = lam*(base_dist.log_prob(noise).mean() + (- sum_log_det - (log_p_x)).mean())**2 + wasserstein_loss(noise,z_k)
        else:
            loss =  (base_dist.log_prob(noise).mean() + (- sum_log_det - (log_p_x)).mean())
        lam = lam 
        loss.backward()
        
        kl_loss += base_dist.log_prob(noise).mean() + (- sum_log_det - (log_p_x)).mean()
        w_loss += wasserstein_loss(noise,z_k)
        loss_1 += loss

        optimizerG.step()

    kl_losses.append(kl_loss.item()/i)
    w_losses.append(w_loss.item()/i)
    losses.append(loss_1.item()/i)
       
    if epoch%1==0:
        lam  = lam * alpha_lam
        print('epoch %d, loss: %f, kl_loss: %f, w_loss: %f'%(epoch,loss.item(),kl_loss/i,w_loss/i))
  return losses, kl_losses, w_losses

def train_AL(netG,optimizerG,data_loader,base_dist,target_dist,lam,rho,lr,num_epochs = 100,ngpu=1):
  device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

  losses = []
  kl_losses = []
  w_losses = []
  # lam = torch.autograd.Variable(-torch.randn(1),requires_grad=True).to(device)
  
  for epoch in range(num_epochs):
    kl_loss = 0.
    w_loss_tmp = 0.
    loss_1 = 0.

    for i, noise in enumerate(data_loader):
        noise=noise.to(device)
        netG.zero_grad()  
        # noise = base_dist.sample((2000,)).to(device)
        z_k, sum_log_det = netG(noise)
        log_p_x = target_dist.log_prob(z_k)
        kl_tmp = (base_dist.log_prob(noise).mean() + (- sum_log_det - (log_p_x)).mean())
        w_loss = wasserstein_loss(noise,z_k)
        loss = wasserstein_loss(noise,z_k) + lam*kl_tmp + 1./2*rho*kl_tmp**2 
        losses.append(loss)
        loss.backward()
        optimizerG.step()
        z_k, sum_log_det = netG(noise)
        log_p_x = target_dist.log_prob(z_k)#.log_prob(z_k)
        kl_tmp = (base_dist.log_prob(noise).mean() + (- sum_log_det - (log_p_x)).mean())
    
        kl_loss += kl_tmp
        w_loss_tmp += wasserstein_loss(noise,z_k)
        loss_1 += loss
    kl_losses.append(kl_loss.item()/i)
    w_losses.append(w_loss_tmp.item()/i)
    losses.append(loss_1.item()/i)

        
        # z_k, sum_log_det = netG(noise)
        # log_p_x = target_dist.log_prob(z_k)#.log_prob(z_k)
        # kl_tmp = (base_dist.log_prob(noise).mean() + (- sum_log_det - (log_p_x)).mean())
        # lam = lam + rho*kl_tmp

    if epoch%1 ==0:
        lam = lam + rho*kl_loss.item()/i
        rho = rho*1.2
        print(lam)
        #   # print('%f'%lam.item())
        # print('epoch %d, lam: %f, loss: %f'%(epoch,lam.item(),loss.item()))
        print('epoch %d, loss: %f, kl_loss:%f, w_loss: %f'%(epoch,loss.item(),kl_tmp.item(),w_loss_tmp.item()/i))
  return losses, kl_losses, w_losses


def train_ADMM(netG1, netG2, optimizerG1, optimizerG2,data_loader, base_dist,target_dist,lamsize,rho,lr,  num_epochs = 100,ngpu=1):
  device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
  l2loss = nn.MSELoss()

  losses_1 = []
  kl_losses1 = []
  w_losses1 = []
  losses_2 = []
  kl_losses2 = []
  w_losses2 = []
  losses = []
  kl_losses = []
  w_losses = []
  kl12_losses =[]
  
  lam = torch.autograd.Variable(torch.ones(lamsize),requires_grad=False).to(device)
  for epoch in range(num_epochs):
    kl_loss_tmp = 0.
    w_loss_tmp = 0.
    kl_loss_tmp1 = 0.
    w_loss_tmp1 = 0.
    loss_tmp = 0.
    kl12_loss_tmp =0.
    for i, noise in enumerate(data_loader):
        netG1.zero_grad()          # noise = base_dist.sample((2000,)).to(device)
        noise = noise.to(device)
        z_k1, sum_log_det1 = netG1(noise)
        z_k2, sum_log_det2 = netG2(noise)
        w_loss = wasserstein_loss(noise,z_k1)
        kl12_loss = torch.mean(lam*(z_k1-z_k2))
        loss_1 = w_loss + 1./2*rho*kl12_loss**2
        loss_1.backward()
        losses_1.append(loss_1.item())
        optimizerG1.step()
        
        # Since parameter netG1 has changes redo 
        netG2.zero_grad()
        z_k2, sum_log_det2 = netG2(noise)
        log_p_x = target_dist.log_prob(z_k2)#.log_prob(z_k)
        loss_2 = (base_dist.log_prob(noise).mean() + (- sum_log_det2 - (log_p_x)).mean()) 
        loss_2.backward()
        losses_2.append(loss_2.item())
      
        optimizerG2.step()

        z_k1, sum_log_det1 = netG1(noise)
        z_k2, sum_log_det2 = netG2(noise)
        kl12_loss_t = l2loss(z_k1,z_k2)

        # lam = lam + rho * kl12_loss.item()

        log_p_x = target_dist.log_prob(z_k2)#.log_prob(z_k)
        # Calculate losses with updated parameter
        loss = w_loss + kl12_loss + 1./2*rho*kl12_loss**2 + (base_dist.log_prob(noise).mean() + (- sum_log_det2 - (log_p_x)).mean())
        losses.append(loss.item())
        log_p_x1 = target_dist.log_prob(z_k1)#.log_prob(z_k)

        kl_loss_tmp1 += (base_dist.log_prob(noise).mean() + (- sum_log_det1 - (log_p_x1)).mean())
        kl_loss_tmp += (base_dist.log_prob(noise).mean() + (- sum_log_det2 - (log_p_x)).mean())
        w_loss_tmp += wasserstein_loss(noise,z_k2)
        w_loss_tmp1 += wasserstein_loss(noise,z_k1)
        kl12_loss_tmp += kl12_loss_t
        loss_tmp += loss
    
    kl12_losses.append(kl12_loss_tmp.item()/i)
    kl_losses.append(kl_loss_tmp.item()/i)
    w_losses.append(w_loss_tmp.item()/i)
    losses.append(loss_tmp.item()/i)
    kl_losses1.append(kl_loss_tmp1.item()/i)
    w_losses1.append(w_loss_tmp1.item()/i)



        # kl_losses.append((base_dist.log_prob(noise).mean() + (- sum_log_det - (log_p_x)).mean()))
        # w_losses.append(wasserstein_loss(noise,z_k))
        # lam = lam #+ 1000*lr*(base_dist.log_prob(noise).mean() + (- sum_log_det - (log_p_x)).mean()).detach()
        # loss.backward()
        # losses.append(loss.item())
        # # scheduler.step(loss)
        
    # optimizerG.step()
    if epoch%1==0:
        lam.data +=  rho * (z_k1-z_k2)
        # rho = rho*1.2
        #   # print('%f'%lam.item())
        # print('epoch %d, lam: %f, loss: %f'%(epoch,lam.item(),loss.item()))
        print('epoch %d, loss: %f, kl_loss:%f, w_loss: %f, kl12_loss:%f, kl1:%f, w1:%f'%(epoch,loss.item(),kl_loss_tmp.item()/i,w_loss_tmp.item()/i,kl12_loss_tmp.item()/i,kl_loss_tmp1.item()/i,w_loss_tmp1.item()/i))
  return losses, kl_losses, w_losses, kl_losses1, w_losses1

# To be deleted
# def train_ADMM(netG1, netG2, optimizerG1, optimizerG2,data_loader, base_dist,target_dist,lam,rho,lr,  num_epochs = 100,ngpu=1):
#   device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#   losses_1 = []
#   kl_losses1 = []
#   w_losses1 = []
#   losses_2 = []
#   kl_losses2 = []
#   w_losses2 = []
#   losses = []
#   kl_losses = []
#   w_losses = []
#   kl12_losses =[]
#   # lam = torch.autograd.Variable(-torch.randn(1),requires_grad=True).to(device)
#   for epoch in range(num_epochs):
#     kl_loss_tmp = 0.
#     w_loss_tmp = 0.
#     loss_tmp = 0.
#     kl12_loss_tmp =0.
#     for i, noise in enumerate(data_loader):
#         netG1.zero_grad()  
#         # noise = base_dist.sample((2000,)).to(device)
#         noise = noise.to(device)
#         z_k1, sum_log_det1 = netG1(noise)
#         z_k2, sum_log_det2 = netG2(noise)
#         w_loss = wasserstein_loss(noise,z_k1)
#         l2loss = nn.MSELoss()
#         kl12_loss = l2loss(z_k1,z_k2)
#         loss_1 = w_loss + lam*kl12_loss + 1./2*rho*kl12_loss**2
#         loss_1.backward()
#         losses_1.append(loss_1.item())
#         optimizerG1.step()
        
#         # Since parameter netG1 has changes redo 
#         netG2.zero_grad()
#         z_k1, sum_log_det1 = netG1(noise)
#         z_k2, sum_log_det2 = netG2(noise)
#         kl12_loss = l2loss(z_k1,z_k2)
#         log_p_x = target_dist.log_prob(z_k2)#.log_prob(z_k)
#         loss_2 = (base_dist.log_prob(noise).mean() + (- sum_log_det2 - (log_p_x)).mean()) + lam*kl12_loss + 1./2*rho*kl12_loss**2 
#         loss_2.backward()
#         losses_2.append(loss_2.item())
      
#         optimizerG2.step()

#         z_k1, sum_log_det1 = netG1(noise)
#         z_k2, sum_log_det2 = netG2(noise)
#         kl12_loss = l2loss(z_k1,z_k2)

#         # lam = lam + rho * kl12_loss.item()

#         log_p_x = target_dist.log_prob(z_k2)#.log_prob(z_k)
#         # Calculate losses with updated parameter
#         loss = w_loss + kl12_loss + 1./2*rho*kl12_loss**2 + (base_dist.log_prob(noise).mean() + (- sum_log_det2 - (log_p_x)).mean())
#         losses.append(loss.item())

#         kl_loss_tmp += (base_dist.log_prob(noise).mean() + (- sum_log_det2 - (log_p_x)).mean())
#         w_loss_tmp += wasserstein_loss(noise,z_k2)
#         kl12_loss_tmp += kl12_loss
#         loss_tmp += loss
    
#     kl12_losses.append(kl12_loss_tmp.item()/i)
#     kl_losses.append(kl_loss_tmp.item()/i)
#     w_losses.append(w_loss_tmp.item()/i)
#     losses.append(loss_tmp.item()/i)



#         # kl_losses.append((base_dist.log_prob(noise).mean() + (- sum_log_det - (log_p_x)).mean()))
#         # w_losses.append(wasserstein_loss(noise,z_k))
#         # lam = lam #+ 1000*lr*(base_dist.log_prob(noise).mean() + (- sum_log_det - (log_p_x)).mean()).detach()
#         # loss.backward()
#         # losses.append(loss.item())
#         # # scheduler.step(loss)
        
#     # optimizerG.step()
#     if epoch%1==0:
#         lam = lam + rho * kl12_loss_tmp.item()/i#kl12_loss.item()
#         rho = rho*1.2
#         #   # print('%f'%lam.item())
#         # print('epoch %d, lam: %f, loss: %f'%(epoch,lam.item(),loss.item()))
#         print('epoch %d, loss: %f, kl_loss:%f, w_loss: %f, kl12_loss:%f'%(epoch,loss.item(),kl_loss_tmp.item()/i,w_loss_tmp.item()/i,kl12_loss_tmp.item()/i))
#   return losses, losses_1, losses_2
