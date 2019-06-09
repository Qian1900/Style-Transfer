#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# ## Image_loader

# In[ ]:


def image_loader(image_name):
    image = Image.open(image_name)
    
    loader = transforms.Compose([
    transforms.Resize((512,512)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# In[ ]:


data_dir = "data/"
content_img = image_loader(data_dir+"geisel.jpg")
style_img = image_loader(data_dir+"14.jpg")


# In[ ]:


assert style_img.size() == content_img.size(),     "we need to import style and content images of the same size"


# ## Image_displayer

# In[ ]:


unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()
def imshow(tensor, ax = plt , title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    ax.imshow(image)
    if title is not None:
      if ax == plt:
        ax.title(title)
      else:
        ax.title.set_text(title)
    ax.axis('off')

plt.figure()

imshow(style_img, title='Style Image')


plt.figure()

imshow(content_img, title='Content Image')


# ## Loss Function

# ### Content Loss

# In[ ]:


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input/2


# ### Style Loss

# In[ ]:


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    #inner product
    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


# In[ ]:


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input/4


# ## Import the model

# In[ ]:


cnn = models.vgg19(pretrained=True).features.to(device).eval()


# In[ ]:


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# In[ ]:


# desired depth layers to compute style/content losses :
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
content_layers_default = ['conv_4']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    print(content_layers)
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            #change MaxPoll to AvgPoll with the same parameters
            #layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:      
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses


# ## Gradient Decent

# In[ ]:


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


# In[ ]:


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1,content_layers=content_layers_default):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img,content_layers=content_layers)
    optimizer = get_input_optimizer(input_img)
    
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            
            style_score = 0
            content_score = 0
            
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
#             if run[0] % 50 == 0:
#                 print("run {}:".format(run))
#                 print('Style Loss : {:4f} Content Loss: {:4f}'.format(
#                     style_score.item(), content_score.item()))

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img


# ## Play with the weighting factors $\alpha/\beta$

# In[ ]:


all_images = []
conv = 'conv_'
style_weights=[2, 3, 4, 5, 6, 7, 8]

for l in range(1, 6):
  content_layers_customized = [conv+str(l)]
  images = []
  for style_weight in style_weights:
      input_img = torch.randn(content_img.data.size(), device=device)
      output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                  content_img, style_img, input_img,
                             style_weight=10**style_weight, content_weight=1, content_layers=content_layers_customized)
      images.append(copy.deepcopy(output))
      plt.figure()
      imshow(output, title='$10^{-%d}$'%style_weight)
  all_images.append(images)


# In[ ]:


# Show the results
cols = ['$10^{-%d}$'%col for col in range(2, 9)]
rows = ['conv_%d'%col for col in range(1, 6)]

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 12))
    
for idx_r, row in enumerate(all_images):
    for idx_c, col in enumerate(row):
      if idx_c >= 4:
        break
      axes[idx_r,idx_c].clear()
      imshow(col, axes[idx_r,idx_c], title="{} {}".format(rows[idx_r], cols[idx_c]))
    
fig.tight_layout()
plt.show()


# ## Change different styles

# In[139]:


files = []
for idx in range(1,20):
  files.append(str(idx)+'.jpg')

content_layers_customized = ['conv_4']
images_dif_styles = []
imgaes_styles = []
for f in files:
    style_img = image_loader(data_dir+f)
    imgaes_styles.append(copy.deepcopy(style_img))
    
    input_img = torch.randn(content_img.data.size(), device=device)
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img,
                           style_weight=10000, content_weight=1, content_layers=content_layers_customized)
    
    images_dif_styles.append(copy.deepcopy(output))
    plt.figure()
    imshow(output)


# In[141]:


# Show the results
fig, axes = plt.subplots(nrows=2*5, ncols=4, figsize=(20, 20))
    
for idx in range(0,19):
  row = idx//4
  col = idx%4
  
  row_o = row*2
  row_t = row*2 + 1
  
  axes[row_o, col].clear()
  imshow(imgaes_styles[idx], axes[row_o, col])
  
  axes[row_t, col].clear()
  imshow(images_dif_styles[idx], axes[row_t, col])
    
fig.tight_layout()
plt.show()


# In[177]:


# Show the final results

chosen = [2,6,8,10,12,15,16,18]

fig, axes = plt.subplots(nrows=2*2, ncols=4, figsize=(10, 10))
    
for idx, chose in enumerate(chosen):
  row = idx//4
  col = idx%4
  
  name = chr(ord('A') + idx)

  row_o = row*2
  row_t = row*2 + 1
  
  axes[row_o, col].clear()

  imshow(imgaes_styles[chose], axes[row_o, col])
  axes[row_o, col].set_title(name +' - 1' )
  
  axes[row_t, col].clear()
  imshow(images_dif_styles[chose], axes[row_t, col])
  axes[row_t, col].set_title(name +' - 2')
    
fig.tight_layout()
plt.show()

