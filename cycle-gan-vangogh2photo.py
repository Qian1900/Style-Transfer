#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from PIL import Image, ImageFile
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


get_ipython().run_line_magic('matplotlib', 'notebook')
import os
import numpy as np

import torch.utils.data as td 
import torchvision as tv

import matplotlib.pyplot as plt
import nntools as nt

import pandas as pd
import json
import random
import pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# ### Image loader

# In[3]:


def image_loader(image_name):
    image = Image.open(image_name)
    
    loader = transforms.Compose([
    transforms.Resize((128,128)),  # scale imported image
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)])  # transform it into a torch tensor

    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# In[4]:


style_img = image_loader("./vangogh2photo/testB/2014-08-02 19:18:15.jpg")
content_img = image_loader("./vangogh2photo/testA/00002.jpg")


# In[5]:


def imshow(image, ax=plt, title=None):
    with torch.no_grad():
        i = image.clone()
        i = i.squeeze(0)
        i = i.to('cpu').numpy()
        i = np.moveaxis(i, [0, 1, 2], [2, 0, 1]) 
        i = (i + 1) / 2
        # clamp the image data to [0,1]
        i[i < 0] = 0
        i[i > 1] = 1 
        ax.figure()
        ax.imshow(i) 
        if title is not None:
            ax.title(title)
        ax.axis('off') 
        ax.pause(0.001)


# In[6]:


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')


# ### Data Loader

# In[7]:


dataset_root_dir = './vangogh2photo'


# In[8]:


class VangoghDataset(td.Dataset):
    
    def __init__(self, root_dir, mode="train", image_size=(128, 128)):
        super(VangoghDataset, self).__init__()
        self.image_size = image_size
        self.mode = mode
        self.images_dir = os.path.join(root_dir, mode+"A")  
        self.data = [f for f in os.listdir(self.images_dir) if os.path.isfile(os.path.join(self.images_dir, f))]
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return "VangoghDataset(mode={}, image_size={})".             format(self.mode, self.image_size)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.data[idx]) 
        img = Image.open(img_path).convert('RGB')
        
        transform = tv.transforms.Compose([
                    tv.transforms.RandomCrop(self.image_size),
                    tv.transforms.RandomHorizontalFlip(),
                    tv.transforms.RandomRotation(30),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize([0.5]*3,[0.5]*3)
                ])
        x = transform(img)
        return x


# In[9]:


vangogh_dataset = VangoghDataset(dataset_root_dir)
vangogh_loader = td.DataLoader(vangogh_dataset, batch_size=1, shuffle=True, pin_memory= True)


# In[10]:


vangogh_dataset.__len__()


# In[11]:


class PhotoDataset(td.Dataset):
    
    def __init__(self, root_dir, mode="train",  image_size=(128, 128)):
        super(PhotoDataset, self).__init__()
        self.image_size = image_size
        self.mode = mode
        self.images_dir = os.path.join(root_dir, mode+"B")  
        self.data = [f for f in os.listdir(self.images_dir) if os.path.isfile(os.path.join(self.images_dir, f))]
        
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return "PhotoDataset(mode={}, image_size={})".             format(self.mode, self.image_size)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.data[idx]) 
        img = Image.open(img_path).convert('RGB')
        
        transform = tv.transforms.Compose([
                    tv.transforms.RandomCrop(self.image_size),
                    tv.transforms.RandomHorizontalFlip(),
                    tv.transforms.RandomRotation(30),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize([0.5]*3,[0.5]*3)
                ])
        x = transform(img)
        return x


# In[12]:


photo_dataset = PhotoDataset(dataset_root_dir)
photo_loader = td.DataLoader(photo_dataset, batch_size=1, shuffle=True, pin_memory= True)


# In[13]:


photo_dataset.__len__()


# ### Discriminator

# In[14]:


class Discriminator(nn.Module):
    """
    70 × 70 PatchGAN 
    """
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = []
        
        in_channels = input_channels
        out_channels = 64
        filter_size = 4
        
        # C64
        self.model.append(nn.Conv2d(in_channels,out_channels, filter_size, stride=2, padding=1))
        self.model.append(nn.LeakyReLU(negative_slope=0.2))
        
        # C128-C256-C512
        for _ in range(3):
            in_channels = out_channels
            out_channels *= 2
            self.model.append(nn.Conv2d(in_channels,out_channels, filter_size, stride=2, padding=1))
            self.model.append(nn.InstanceNorm2d(out_channels))
            self.model.append(nn.LeakyReLU(negative_slope=0.2))
            
        self.model.append(nn.ZeroPad2d((1, 0, 1, 0)))
        # After the last layer, we apply a convolution to produce a 1-dimensional output.
        self.model.append(nn.Conv2d(out_channels, 1, filter_size, padding=1))
        
        self.model = nn.Sequential(*self.model)
        
    def forward(self, x): 
        y = self.model(x)
        return y
    
    def criterion(self, dy, dgx):
        valid_y = torch.Tensor(np.ones(dy.shape)).to(device)
        l_1 = nn.functional.mse_loss(dy, valid_y)
        
        valid_yfX = Tensor(np.zeros(dgx.shape)).to(device)
        l_2 = nn.functional.mse_loss(dgx, valid_yfX)
        
        loss = l_1 + l_2
        return loss


# ### Generator

# In[15]:


class ResidualBlock(nn.Module):
    """
    define residual block
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        self.model = []
        
        self.model.append(nn.ReflectionPad2d(1))
        self.model.append(nn.Conv2d(in_channels,in_channels, 3))
        self.model.append(nn.InstanceNorm2d(in_channels))
        self.model.append(nn.ReLU())
        
        self.model.append(nn.ReflectionPad2d(1))
        self.model.append(nn.Conv2d(in_channels,in_channels, 3))
        self.model.append(nn.InstanceNorm2d(in_channels))
        self.model = nn.Sequential(*self.model)
        
    def forward(self, x):
#         return nn.ReLU(x + self.model(x))
        return x + self.model(x)


# In[16]:


class Generator(nn.Module):
    """
    6 residual blocks for 128 × 128 training images
    """
    def __init__(self, input_channels):
        super(Generator, self).__init__()
        self.model = []
        
        in_channels = input_channels
        out_channels = 64
        
        # c7s1-64
        self.model.append(nn.ReflectionPad2d(in_channels))
        self.model.append(nn.Conv2d(in_channels,out_channels, 7))
        self.model.append(nn.InstanceNorm2d(out_channels))
        self.model.append(nn.ReLU())
        
        # d128, d256
        for _ in range(2):
            in_channels = out_channels
            out_channels *= 2
            self.model.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
            self.model.append(nn.InstanceNorm2d(out_channels))
            self.model.append(nn.ReLU())
            
        # R256 * 6
        in_channels = out_channels # 256
        for _ in range(6):
            self.model.append(ResidualBlock(in_channels))
            
        # u128, u64
        for _ in range(2):
            out_channels //= 2
            self.model.append(nn.Upsample(scale_factor=2))
            self.model.append(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
            self.model.append(nn.InstanceNorm2d(out_channels))
            self.model.append(nn.ReLU())
            in_channels = out_channels
        
        # c7s1-3
        out_channels = 3
        self.model.append(nn.ReflectionPad2d(input_channels))
        self.model.append(nn.Conv2d(in_channels,out_channels, 7))
        self.model.append(nn.Tanh())
        
        self.model = nn.Sequential(*self.model)
         
    def forward(self, x): 
        y = self.model(x)
        return y


# ### Loss

# In[17]:


def AdversarialLoss(D, fake):
    """
    compute the adversarial loss L_GAN(G, DY , X, Y ) 
    """
    res = D(fake)
    valid = Tensor(np.ones(res.shape)).to(device)
    loss = nn.functional.mse_loss(res, valid)
    return loss


# In[18]:


def CycleConsistencyLoss(G, F, fake_X, fake_Y, X, Y):
    """
    compute the cycle consistency loss L_cyc (G, F ) 
    """
    x_cycled = F(fake_Y)
    y_cycled = G(fake_X)
    loss1 = nn.functional.l1_loss(x_cycled, X)
    loss2 = nn.functional.l1_loss(y_cycled, Y)
    loss = loss1+loss2
    return loss/2


# In[19]:


def fullLoss(G, F, D_X, D_Y, X, Y, idx, lam = 10.0):
    """
    compute the loss for cycle Gan
    """
    fake_X = F(Y)
    fake_Y = G(X)
    l_gan1 = AdversarialLoss(D_Y, fake_Y)
    l_gan2 = AdversarialLoss(D_X, fake_X)
    l_cyc = CycleConsistencyLoss(G, F, fake_X, fake_Y, X, Y)
    loss = l_gan1 + l_gan2 + lam * l_cyc
    
    if idx%50 == 0:
        print('Advers_loss_Dy_Gx: %.3f Advers_loss_Dx_Fy: %.3f Cyc_loss: %.3f' %
                            (l_gan1, l_gan2, l_cyc))
    plt_Advers_loss_Dy_Gx.append(l_gan1)
    plt_Advers_loss_Dx_Fy.append(l_gan2)
    plt_Cyc_loss.append(l_cyc)
    plt_loss.append(loss)
    return loss


# ### Image Buffer (reduce model ocillation)

# In[20]:


class ImageHistoryBuffer(object):
    def __init__(self, max_size = 50):
        """
        :param max_size: Maximum number of images that can be stored in the image history buffer.
        """
        self.max_size = max_size
        self.image_history_buffer = []

    def get_from_image_history_buffer(self, img):
        """
        Get a random sample of images from the history buffer.
        """
        if len(self.image_history_buffer) < self.max_size:
            self.image_history_buffer.append(img)
            return img
        else:
            if random.uniform(0, 1) > 0.5:
                i = random.randint(0, self.max_size - 1)
                tmp = self.image_history_buffer[i].clone()
                self.image_history_buffer[i] = img
                return tmp
            else:
                return img


# ### Plot Losses

# In[21]:


# loss ploting interval
interval = min(vangogh_dataset.__len__(), photo_dataset.__len__()) 

def show_loss():
    plt.xlabel("# of epochs")
    plt.ylabel("Adversarial_loss_Dy_Gx")
    plt.plot(plt_Advers_loss_Dy_Gx[0:len(plt_Advers_loss_Dy_Gx):interval], label = 'Adversarial_loss_Dy_Gx')
    plt.legend()
    plt.show()
    plt.pause(0.0001)
    plt.xlabel("# of epochs")
    plt.ylabel("Adversarial_loss_Dx_Fy")
    plt.plot(plt_Advers_loss_Dx_Fy[0:len(plt_Advers_loss_Dx_Fy):interval], label = 'Adversarial_loss_Dx_Fy')
    plt.legend()
    plt.show()
    plt.pause(0.0001)
    plt.xlabel("# of epochs")
    plt.ylabel("Cycle_loss")
    plt.plot(plt_Cyc_loss[0:len(plt_Cyc_loss):interval], label = 'Cycle_loss')
    plt.legend()
    plt.show()
    plt.pause(0.0001)
    plt.xlabel("# of epochs")
    plt.ylabel("Full_loss")
    plt.plot(plt_loss[0:len(plt_loss):interval], label = 'Full_loss')
    plt.legend()
    plt.show()
    plt.pause(0.0001)


# ### trainning process

# In[22]:


def backprop_deep(G, F, D_X, D_Y, content_loader, style_loader, start=0, T=200, gamma=0.0002):
    params = list(G.parameters()) + list(F.parameters())
    optimizer_Ge = torch.optim.Adam(params, lr=gamma, betas=(0.5, 0.999))
    optimizer_Dx = torch.optim.Adam(D_X.parameters(), lr=gamma, betas=(0.5, 0.999))
    optimizer_Dy = torch.optim.Adam(D_Y.parameters(), lr=gamma, betas=(0.5, 0.999))
    
    buffer_X_fromY = ImageHistoryBuffer()
    buffer_Y_fromX = ImageHistoryBuffer()
    
    for epoch in range(start, T): 
        #  linearly decay the rate to zero over the next 100 epochs. 
        if epoch >= 100:
            for g in optimizer_Ge.param_groups:
                g['lr'] = gamma - gamma/100 * (epoch-100+1)
            for g in optimizer_Dx.param_groups:
                g['lr'] = gamma -  gamma/100 * (epoch-100+1)
            for g in optimizer_Dy.param_groups:
                g['lr'] = gamma - gamma/100 * (epoch-100+1)
               
        for idx, img in enumerate(zip(content_loader, style_loader)):
            X = img[0].to(device)
            Y = img[1].to(device)
            # Generators
            # Initialize the gradients to zero
            optimizer_Ge.zero_grad()
            # Forward propagation and Error evaluation
            loss_Ge = fullLoss(G, F, D_X, D_Y, X, Y, idx)
            # Back propagation
            loss_Ge.backward()
            # Parameter update
            optimizer_Ge.step()

            # Discriminator X
            # Initialize the gradients to zero
            optimizer_Dx.zero_grad()
            # Forward propagation and Error evaluation
            dx = D_X(X)
            # To reduce model oscillation, using a history of generated images rather than
            # the ones produced by the latest generators. 
            dfy = buffer_X_fromY.get_from_image_history_buffer(D_X(F(Y)))
            loss_Dx = D_X.criterion(dx, dfy)/2
            # Back propagation
            loss_Dx.backward(retain_graph=True)
            # Parameter update
            optimizer_Dx.step()

            # Discriminator Y
            # Initialize the gradients to zero
            optimizer_Dy.zero_grad()
            # Forward propagation and Error evaluation
            dy = D_Y(Y)
            dgx = buffer_X_fromY.get_from_image_history_buffer(D_Y(G(X)))
            loss_Dy = D_Y.criterion(dy, dgx)/2
            # Back propagation
            loss_Dy.backward(retain_graph=True)
            # Parameter update
            optimizer_Dy.step()
            
            
            if idx%50 == 0:
                print('[%d, %d] G_loss: %.3f Dx_loss: %.3f Dy_loss: %.3f\n' %
                        (epoch, idx, loss_Ge.item(), loss_Dx.item(), loss_Dy.item()))
            
            if idx%100 == 0:
                fake = G(content_img)
                imshow(fake, title='G(content_img) [%d, %d]'%(epoch, idx))
                
                cycled = F(fake)
                imshow(cycled, title='F(G(content_img)) [%d, %d]'%(epoch, idx))
                
        show_loss()
    
        # save model checkpoint
        torch.save({
            'G_state_dict': G.state_dict(),
            'F_state_dict': F.state_dict(),
            'Dx_state_dict':D_X.state_dict(),
            'Dy_state_dict':D_Y.state_dict(),
            }, saving_dir+"%d.pth" % (epoch))
        
        # save losses
        saved_losses = {"a":plt_Advers_loss_Dy_Gx, "b":plt_Advers_loss_Dx_Fy, "c":plt_Cyc_loss, "d":plt_loss}
        with open(saving_dir+'loss.backup.epoch_%d' % (epoch),'wb') as backup_file:
            pickle.dump(saved_losses, backup_file)

    return G, F, D_X, D_Y


# ### Generate model

# In[23]:


channels = 3
G = Generator(channels).to(device)
F = Generator(channels).to(device)
D_X = Discriminator(channels).to(device)
D_Y = Discriminator(channels).to(device)


# ### Load model if necessary

# In[24]:


def init_weights(m):
    """
    initial the weights using Gaussian distribution 
    """
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        m.bias.data.fill_(0.0)


# In[25]:


#Flag = False
#epoch = -1
saving_dir = 'cycle_gan_vangogh2photo/'
Flag = True
epoch = 168
if Flag:
    # Load pretrained models
    checkpoint = torch.load(saving_dir+"%d.pth" % (epoch))
    G.load_state_dict(checkpoint['G_state_dict'])
    F.load_state_dict(checkpoint['F_state_dict'])
    D_X.load_state_dict(checkpoint['Dx_state_dict'])
    D_Y.load_state_dict(checkpoint['Dy_state_dict'])
    # load losses
    with open(saving_dir+'loss.backup.epoch_%d' % (epoch), 'rb') as backup_file:
        tmp = pickle.load(backup_file)
        plt_Advers_loss_Dy_Gx = tmp["a"]
        plt_Advers_loss_Dx_Fy = tmp["b"]
        plt_Cyc_loss = tmp["c"]
        plt_loss = tmp["d"]
#         plt_Advers_loss_Dy_Gx, plt_Advers_loss_Dx_Fy, plt_Cyc_loss, plt_loss = pickle.load( backup_file)
else:
    # Initialize weights
    G.apply(init_weights)
    F.apply(init_weights)
    D_X.apply(init_weights)
    D_Y.apply(init_weights)
    os.makedirs(saving_dir, exist_ok=True)
    # store the loss
    plt_Advers_loss_Dy_Gx = []
    plt_Advers_loss_Dx_Fy = []
    plt_Cyc_loss = []
    plt_loss = []


# ### Train

# In[26]:


backprop_deep(G, F, D_X, D_Y, vangogh_loader, photo_loader, start=epoch+1)


# In[27]:


saving_dir = 'cycle_gan_vangogh2photo/final_model_'
torch.save(G, saving_dir+'G')
torch.save(F, saving_dir+'F')

