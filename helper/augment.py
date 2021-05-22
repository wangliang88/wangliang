import random
import numpy as np
import torch
import cv2
from torchvision import transforms

w_jitter = transforms.ColorJitter(brightness=0.00001, contrast=0.00001, saturation=0.00001, hue=0.00001)
s_jitter = transforms.ColorJitter(brightness=0.00003, contrast=0.00003, saturation=0.00003, hue=0.00003)
img_PIL = transforms.ToPILImage()
def w_blur(image):
    # Gaussian Blud (sigma between 0 and 1.4)
    sigma = random.random() /5
    ksize = int(3 * sigma)
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    return image
def s_blur(image):
    # Gaussian Blud (sigma between 0 and 1.1)
    sigma = random.random() /10
    ksize = int(3 * sigma)
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    return image
    
def strong_aug(image):
    image = image.numpy()
    for i in range(image.shape[0]):
        if random.random() > 0.8 :
            image[i,:,:,:] = s_blur(image[i,:,:,:])
        if random.random() > 0.8 :
            temp = img_PIL(image[i,:,:,:].astype('uint8').transpose(1,2,0))
            image[i,:,:,:] = np.array(s_jitter(temp)).transpose(2,0,1)
    image = torch.from_numpy(image).float()
    return image
    
def weak_aug(image):
    image = image.numpy()
    for i in range(image.shape[0]):
        if random.random() > 0.8 :
            image[i,:,:,:] = w_blur(image[i,:,:,:])
        if random.random() > 0.8 :
            temp = img_PIL(image[i,:,:,:].astype('uint8').transpose(1,2,0))
            image[i,:,:,:] = np.array(w_jitter(temp)).transpose(2,0,1)
    image = torch.from_numpy(image).float()
    return image  
