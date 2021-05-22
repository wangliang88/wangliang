from networks.GAN_Unet import GAN,FCDiscriminator
import torch
import random
import numpy as np
from torch import nn as nn
from tqdm import trange
from helper.ema import EMA as ema
from typing import Callable
from typing import Dict
from typing import Sequence
from helper.utils import EasyDict
from helper.visual_helper import VisualHelper
from helper.model_saver import ModelSaver
from helper.optim import Optimizer
from data_provider.sod_dataloader import SOD_Dataloader
import torch.nn.functional as F
LOSS_FUNC = Callable[[torch.Tensor, torch.Tensor or float, torch.Tensor or None], torch.Tensor]
WEIGHT_INIT_FUNC = Callable[[GAN], None]

LOSSES_TYPE = Sequence[Dict[str, LOSS_FUNC]]

__all__ = ['Trainer']

def sigmoid_rampup(current,rampup_length):
     phase = 1.0 - current / rampup_length
     return float(np.exp(-5.0 * phase * phase))
     
class TrainDataWrapper(EasyDict):
    def __init__( self, dataloader: SOD_Dataloader, \
                  lambda_adv: float or None = None, lambda_sal: float or None = 1.0, \
                  mask_T: float or None = None, start_time: int or None = None, *args, **kwargs ):
        super().__init__(*args, **kwargs)
        self.dataloader: SOD_Dataloader = dataloader
        self.lambda_sal: float or None = lambda_sal
        self.lambda_adv: float or None = lambda_adv
        self.start_time: float or None = start_time
        self.mask_T: float or None = mask_T


class Trainer(object): 
    """
    the helper to train network model
    """
    def __init__( self,
                  train_fullsupervised_data: TrainDataWrapper,
                  train_semi_data: TrainDataWrapper,
                  train_gt_data: TrainDataWrapper,
                  adv_loss_function: LOSS_FUNC,
                  sal_loss_function: LOSS_FUNC,
                  mse_loss_function: LOSS_FUNC,
                  max_iter_time: int,
                  ignore_value: float,
                  generator_optim_create_func_1: Optimizer,
                  generator_optim_create_func_2: Optimizer,
                  discriminator_optim_create_func: Optimizer,
                  generate_1_lr: float,
                  generate_2_lr: float,
                  discriminator_lr: float,
                  device: torch.device = torch.device('cpu'),
                  pretrained_model_path: str = None,
                  visual_helper: VisualHelper = None,
                  model_saver: ModelSaver = None,
                  g_weight_init_func: WEIGHT_INIT_FUNC = None,
                  d_weight_init_func: WEIGHT_INIT_FUNC = None,
                  is_use_grab: bool = False,
                  *args,
                  **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)
        self.model:GAN = GAN()
        self.model.train()
        if g_weight_init_func is not None:
            self.model.generator_1.apply(g_weight_init_func)
            self.model.generator_2.apply(g_weight_init_func)
        if d_weight_init_func is not None:
            self.model.discriminator.apply(d_weight_init_func)
        if pretrained_model_path is not None:
            self.model.generator_1.load_weight(pretrained_model_path)
            self.model.generator_2.load_weight(pretrained_model_path)
         
        self.model.to(device)
        self.max_iter_time = max_iter_time
        generator_optim_create_func_1(self.model.generator_1, generate_1_lr)
        generator_optim_create_func_2(self.model.generator_2, generate_2_lr)
        self.G_optim_1 = generator_optim_create_func_1
        self.G_optim_2 = generator_optim_create_func_2
        discriminator_optim_create_func(self.model.discriminator, discriminator_lr)
        self.D_optim = discriminator_optim_create_func
        self.device = device
        self.visual_helper = visual_helper
        self.model_saver = model_saver
        self.train_fullsupervised_data = train_fullsupervised_data
        self.train_semi_data = train_semi_data
        self.train_gt_data = train_gt_data
        self.adv_loss_function = adv_loss_function
        self.sal_loss_function = sal_loss_function
        self.up_loss = mse_loss_function
        self.teacher_loss = nn.MSELoss()
        self.fake_label = 0.0
        self.real_label = 1.0
        self.ignore_value = ignore_value
        self.is_use_grab: bool = is_use_grab

    def get_data( self, datawrapper: TrainDataWrapper ):
        return next(datawrapper.dataloader)

    def train( self ):
        if self.visual_helper is not None:
            loss_list = list()
        train_full = self.train_fullsupervised_data 
        train_semi = self.train_semi_data
        fake_with_gt_image = None
        if train_semi.lambda_sal:
            assert train_semi.mask_T is not None, "the mask_T of train_semi_data should not be None"
        for iter_time in trange(1, self.max_iter_time + 1):
            fake_image_list = list() #output of saliency network 
            loss_dict = dict()
            # Train labeled data
            self.model.discriminator.freeze_bp(True)#freeze the discriminator backwards
            train_full_images = None
            train_full_labels = None
            if train_full is not None:
                item = self.get_data(train_full)
                if train_full.lambda_sal:
                    train_full_images = item['image'].to(self.device)
                    train_full_labels = item['label'].to(self.device)
                    train_full_labels.unsqueeze_(dim=1)
                    sal_pred_on_train_full_images_1 = self.model.forward_G1(train_full_images)
                    sal_pred_on_train_full_images_2 = self.model.forward_G2(train_full_images)
                    train_full_sal_loss_1 = self.sal_loss_function(sal_pred_on_train_full_images_1, \
                                                                 train_full_labels)
                    train_full_sal_loss_2 = self.sal_loss_function(sal_pred_on_train_full_images_2, \
                                                                 train_full_labels)
                    loss_dict['train_full_sal_loss_1'] = train_full_sal_loss_1.item()
                    loss_dict['train_full_sal_loss_2'] = train_full_sal_loss_2.item()
                    train_full_sal_loss_1 = train_full.lambda_sal * train_full_sal_loss_1
                    train_full_sal_loss_2 = train_full.lambda_sal * train_full_sal_loss_1
                    (train_full_sal_loss_1 + train_full_sal_loss_2).backward() #at the begin use the image with label to pretrain the generater-1
                    
                if train_full.lambda_adv:
                    if train_full_images is None or train_full_labels is None:
                        train_full_images = item['image'].to(self.device)
                        train_full_labels = item['label'].to(self.device)
                        train_full_labels.unsqueeze_(dim=1)
                    sal_pred_on_train_full_images_1 = self.model.forward_G1(train_full_images)
                    sal_pred_on_train_full_images_2 = self.model.forward_G2(train_full_images)
                    sal_pred_on_train_full_images_sigmoid_1 = torch.sigmoid(sal_pred_on_train_full_images_1)
                    sal_pred_on_train_full_images_sigmoid_2 = torch.sigmoid(sal_pred_on_train_full_images_2)
                    d_out_1,fo_d_out_1 = self.model.forward_D(sal_pred_on_train_full_images_sigmoid_1)
                    d_out_2,fo_d_out_2 = self.model.forward_D(sal_pred_on_train_full_images_sigmoid_2)
                    fake_image_list.append(sal_pred_on_train_full_images_sigmoid_1.detach()) #the labeled image of the generator-1 output
                    fake_image_list.append(sal_pred_on_train_full_images_sigmoid_2.detach()) #the labeled image of the generator-2 output
                    fake_with_gt_image_1 = fake_image_list[0]
                    fake_with_gt_image_2 = fake_image_list[1]
                    train_full_adv_loss_1 = self.adv_loss_function(d_out_1, self.real_label)#train the generater
                    train_full_adv_loss_2 = self.adv_loss_function(d_out_2, self.real_label)#train the generater
                    # train_full_adv_loss += self.adv_loss_function(fo_d_out, self.real_label)
                    del fo_d_out_1
                    del fo_d_out_2
                    loss_dict['train_full_adv_loss_1'] = train_full_adv_loss_1.item()
                    loss_dict['train_full_adv_loss_2'] = train_full_adv_loss_2.item()
                    train_full_adv_loss = train_full.lambda_adv * ( train_full_adv_loss_1 + train_full_adv_loss_2 )
                    train_full_adv_loss.backward()

            # Training semi-supervised data
            train_semi_images = None
            if train_semi is not None:
                item = self.get_data(train_semi)
                if train_semi.lambda_sal and (train_semi.start_time is None or iter_time > train_semi.start_time):
                    train_semi_images = item['image'].to(self.device)
                    sal_pred_on_train_semi_images_1 = self.model.forward_G1(train_semi_images)#shengchengqi biaoqian
                    sal_pred_on_train_semi_images_sigmoid_1 = torch.sigmoid(sal_pred_on_train_semi_images_1.detach())
                    d_out_1, fo_d_out_1 = self.model.forward_D(sal_pred_on_train_semi_images_sigmoid_1)
                    sal_fake_label_on_train_semi_images_1 = sal_pred_on_train_semi_images_sigmoid_1.gt_(0.5).float()#the wei biaoqian, binary the output of generator ,the mutiply the v 
                    weight_1 = torch.sigmoid_(fo_d_out_1)
                    if train_semi.lambda_adv:
                        ignore_index_1 = weight_1 < train_semi.mask_T
                        sal_fake_label_on_train_semi_images_1[ignore_index_1] = self.ignore_value
                        del ignore_index_1
                    train_semi_sal_loss_1 = self.sal_loss_function(sal_pred_on_train_semi_images_1, \
                                                             sal_fake_label_on_train_semi_images_1)#train the generator with v
                    loss_dict['train_semi_sal_loss_1'] = train_semi_sal_loss_1.item()
                    
                    # generator-2  network
                    sal_pred_on_train_semi_images_2 = self.model.forward_G2(train_semi_images)#shengchengqi biaoqian
                    sal_pred_on_train_semi_images_sigmoid_2 = torch.sigmoid(sal_pred_on_train_semi_images_2.detach())
                    d_out_2, fo_d_out_2 = self.model.forward_D(sal_pred_on_train_semi_images_sigmoid_2)
                    sal_fake_label_on_train_semi_images_2 = sal_pred_on_train_semi_images_sigmoid_2.gt_(0.5).float()#the wei biaoqian, binary the output of generator ,the mutiply the v 
                    weight_2 = torch.sigmoid_(fo_d_out_2)
                    if train_semi.lambda_adv:
                        ignore_index_2 = weight_2 < train_semi.mask_T
                        sal_fake_label_on_train_semi_images_2[ignore_index_2] = self.ignore_value
                        del ignore_index_2
                    train_semi_sal_loss_2 = self.sal_loss_function(sal_pred_on_train_semi_images_2, \
                                                             sal_fake_label_on_train_semi_images_2)#train the generator with v
                    loss_dict['train_semi_sal_loss_2'] = train_semi_sal_loss_2.item()
                    train_semi_sal_loss = train_semi.lambda_sal * (train_semi_sal_loss_1 + train_semi_sal_loss_2)
                    if iter_time % 2 == 0:
                        index_1_up = fo_d_out_1 > fo_d_out_2 
                        sal_pred_on_train_semi_images_2[index_1_up] = self.ignore_value
                        _gt02_index = fo_d_out_2 > 0.2
                        sal_pred_on_train_semi_images_2[_gt02_index] = self.ignore_value
                        T_up_loss = self.up_loss(sal_pred_on_train_semi_images_1,sal_pred_on_train_semi_images_2.detach()) 
                    if iter_time % 2 == 1:
                        index_2_up =   fo_d_out_1 < fo_d_out_2
                        sal_pred_on_train_semi_images_1[index_2_up] = self.ignore_value
                        _gt02_index = fo_d_out_1 > 0.2
                        sal_pred_on_train_semi_images_1[_gt02_index] = self.ignore_value
                        T_up_loss = self.up_loss(sal_pred_on_train_semi_images_2,sal_pred_on_train_semi_images_1.detach())
                    
                    (train_semi_sal_loss + train_semi.lambda_sal*0.001*T_up_loss).backward()
                    del sal_fake_label_on_train_semi_images_1
                    del sal_pred_on_train_semi_images_sigmoid_1
                    del sal_pred_on_train_semi_images_1
                    del sal_fake_label_on_train_semi_images_2
                    del sal_pred_on_train_semi_images_sigmoid_2
                    del sal_pred_on_train_semi_images_2
                    del d_out_1
                    del d_out_2
                    del fo_d_out_1
                    del fo_d_out_2
                    del weight_1
                    del weight_2
                    
                    
                if train_semi.lambda_adv:
                    if train_semi_images is None:
                        train_semi_images = item['image'].to(self.device)
                    sal_pred_on_train_semi_images_1 = self.model.forward_G1(train_semi_images)
                    sal_fake_label_on_train_semi_images_1 = torch.sigmoid(sal_pred_on_train_semi_images_1)
                    d_out_1,fo_d_out_1 = self.model.forward_D(sal_fake_label_on_train_semi_images_1)
                    fake_image_list.append(sal_fake_label_on_train_semi_images_1.detach())#train generator with unlabelled image
                    
                    sal_pred_on_train_semi_images_2 = self.model.forward_G2(train_semi_images)
                    sal_fake_label_on_train_semi_images_2 = torch.sigmoid(sal_pred_on_train_semi_images_2)
                    d_out_2,fo_d_out_2 = self.model.forward_D(sal_fake_label_on_train_semi_images_2)
                    fake_image_list.append(sal_fake_label_on_train_semi_images_2.detach())#train generator with unlabelled image
                    
                    train_semi_adv_loss_1 = self.adv_loss_function(d_out_1, self.real_label)
                    train_semi_adv_loss_2 = self.adv_loss_function(d_out_2, self.real_label)
                    # train_semi_adv_loss += self.adv_loss_function(fo_d_out, self.real_label)
                    del fo_d_out_1
                    del fo_d_out_2
                    loss_dict['train_semi_adv_loss_1'] = train_semi_adv_loss_1.item()
                    loss_dict['train_semi_adv_loss_2'] = train_semi_adv_loss_2.item()
                    train_semi_adv_loss = train_semi.lambda_adv * (train_semi_adv_loss_1 + train_semi_adv_loss_2)
                    train_semi_adv_loss.backward()

            # train D
            fake_image = None
            if len(fake_image_list) == 0:
                pass
            elif len(fake_image_list) == 1:
                fake_image = fake_image_list[0]
            elif len(fake_image_list) == 2:
                fake_image = torch.cat(fake_image_list,dim=0)
            elif len(fake_image_list) == 3:
                fake_image = torch.cat(fake_image_list,dim=0)
            elif len(fake_image_list) == 4:
                fake_image = torch.cat(fake_image_list,dim=0)

            if fake_image is not None and self.train_gt_data is not None:
                self.model.discriminator.freeze_bp(False)
                if fake_with_gt_image_1 is not None and fake_with_gt_image_2 is not None:
                    self.model.discriminator.freeze_encoder_bp(True)
                    d_out_fake_1,fo_d_out_fake_1 = self.model.forward_D(fake_image_list[0])
                    d_out_label_1 = 1 - torch.abs(fake_with_gt_image_1 - train_full_labels)
                    d_out_loss_on_fake_1 = self.adv_loss_function(fo_d_out_fake_1[:d_out_label_1.size(0)],d_out_label_1.detach())# update the generator to train v
                    d_out_fake_2,fo_d_out_fake_2 = self.model.forward_D(fake_image_list[1])
                    d_out_label_2 = 1 - torch.abs(fake_with_gt_image_2 - train_full_labels)
                    d_out_loss_on_fake_2 = self.adv_loss_function(fo_d_out_fake_2[:d_out_label_2.size(0)],d_out_label_2.detach())# update the generator to train v
                    (0.5*(d_out_loss_on_fake_1 + d_out_loss_on_fake_2)).backward()  
                    if len(fake_image_list) == 4 and iter_time > train_semi.start_time:
                        if iter_time == train_semi.start_time + 1:
                            self.teacher = ema(self.model.discriminator,0.99)  #teacher net of discriminator
                        self.teacher.update(self.model.discriminator)
                        _,student_out_1 = self.model.forward_D(fake_image_list[2]) 
                        _,teacher_out_1 = self.teacher.ema(fake_image_list[2])
                        teacher_consistency = 0.5*0.001*sigmoid_rampup(iter_time ,self.max_iter_time )
                        mean_teacher_loss_1 = teacher_consistency *  self.teacher_loss(student_out_1,teacher_out_1.detach())/8.0#mse loss
                        #mean_teacher_loss_1.backward()
                        _,student_out_2 = self.model.forward_D(fake_image_list[3]) 
                        _,teacher_out_2 = self.teacher.ema(fake_image_list[3])
                        #teacher_consistency = 0.0001*sigmoid_rampup(iter_time ,self.max_iter_time )
                        mean_teacher_loss_2 = teacher_consistency *  self.teacher_loss(student_out_2,teacher_out_2.detach())/8.0#mse loss
                        #mean_teacher_loss_2.backward()
                        if mean_teacher_loss_2 is not None and mean_teacher_loss_2 is not None :
                            #print(mean_teacher_loss_1.data)
                            (mean_teacher_loss_1 + mean_teacher_loss_2).backward()
                        
                    self.model.discriminator.freeze_encoder_bp(False)
                d_out_fake,fo_d_out_fake = self.model.forward_D(fake_image)
                d_out_loss_on_fake = self.adv_loss_function(d_out_fake,self.fake_label)
                (0.5*d_out_loss_on_fake).backward()
                loss_dict['d_out_loss_on_fake'] = d_out_loss_on_fake.item()
                item = self.get_data(self.train_gt_data)
                real_label = item['label'].to(self.device)
                real_label.unsqueeze_(dim=1)
                d_out_true,fo_d_out_true = self.model.forward_D(real_label)
                d_out_loss_on_true = self.adv_loss_function(d_out_true, self.real_label)
                # d_out_loss_on_true += self.adv_loss_function(fo_d_out_true, self.real_label)
                d_out_loss_on_true.backward()
                loss_dict['d_out_loss_on_true'] = d_out_loss_on_true.item()
                self.D_optim.step()
                
            self.G_optim_1.step()
            self.G_optim_2.step()
            if self.visual_helper is not None:
                loss_list.append(loss_dict)
                self.visual_helper.add_timer()
                if self.visual_helper.is_catch_snapshot():
                    info_dict = dict()
                    counter_dict = dict()
                    for item in loss_list:
                        for key,value in item.items():
                            info_dict[key] = info_dict.get(key,0.0) + info_dict[key]
                            counter_dict[key] = info_dict.get(key,0.0) + 1.0
                    for key,value in info_dict.items():
                        info_dict[key] /= counter_dict[key]
                    self.visual_helper(0,info_dict,None)
                    loss_list.clear()
            if self.model_saver is not None:
                self.model_saver(self.model.generator_1, save_base_name='G1')
              #  self.model_saver(self.model.generator_2, save_base_name='G2')
                self.model_saver(self.model.discriminator, save_base_name='D')

        if self.model_saver is not None:
                self.model_saver(self.model.generator_1, isFinal=True, save_base_name='G1')
              #  self.model_saver(self.model.generator_2, isFinal=True, save_base_name='G2')
                self.model_saver(self.model.discriminator, isFinal=True, save_base_name='D')
        if self.visual_helper is not None:
            self.visual_helper.close()
