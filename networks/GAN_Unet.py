from networks.Generator import Generator
from networks.UNet_D import FCDiscriminator
import torch
import torch.nn as nn

class GAN(object):

    def __init__(self,device=torch.device('cpu'),G_pth_path=None):
        self.generator_1:Generator = Generator()
        self.generator_2:Generator = Generator()
        self.discriminator = FCDiscriminator()
        if G_pth_path is not None:
            self._load_generator_weight(G_pth_path)
        self.device = device
        self.generator_1.to(device)
        self.generator_2.to(device)
        self.discriminator.to(device)
        self.train()

    def train(self):
        self.generator_1.train()
        self.generator_2.train()
        self.discriminator.train()
        return self

    def eval(self):
        self.generator_1.eval()
        self.generator_2.eval()
        self.discriminator.eval()
        return self

    def to(self,*args,**kwargs):
        self.discriminator.to(*args,**kwargs)
        self.generator_1.to(*args,**kwargs)
        self.generator_2.to(*args,**kwargs)
        return self

    def _load_generator_weight(self, pth_path):
        return self.generator_1.load_weight(pth_path) and self.generator_2.load_weight(pth_path)
        

    def forward_G1(self,x):
        return self.generator_1(x)
 
 
    def forward_G2(self,x):
        return self.generator_2(x)
        
        
    def forward_D(self,x,is_only_encoder = False):
        return self.discriminator(x,is_only_encoder)


    def apply(self,callback):
        self.discriminator.apply(callback)
        self.generator_1.apply(callback)
        self.generator_2.apply(callback)