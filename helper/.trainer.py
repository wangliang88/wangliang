from networks.GAN_Unet import GAN,FCDiscriminator
import torch
import numpy as np
from torch import nn as nn
from helper.ema import EMA as ema
from tqdm import trange
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
                  max_iter_time: int,
                  ignore_value: float,
                  generator_optim_create_func: Optimizer,
                  discriminator_optim_create_func: Optimizer,
                  generate_lr: float,
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
            self.model.generator.apply(g_weight_init_func)
        if d_weight_init_func is not None:
            self.model.discriminator.apply(d_weight_init_func) 
        if pretrained_model_path is not None:
            self.model.generator.load_weight(pretrained_model_path)
        #self.teacher_generator = ema(self.model.generator,0.99)
        self.teacher_discriminator = ema(self.model.discriminator,0.99)  #teacher net 
        self.model.to(device)
        self.max_iter_time = max_iter_time
        generator_optim_create_func(self.model.generator, generate_lr)
        self.G_optim = generator_optim_create_func
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
        self.teacher_discriminator_loss = nn.MSELoss()
        self.teacher_generator_loss = nn.MSELoss()
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
            fake_image_list = list()
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
                    sal_pred_on_train_full_images = self.model.forward_G(train_full_images)
                    train_full_sal_loss = self.sal_loss_function(sal_pred_on_train_full_images, \
                                                                 train_full_labels)
                    loss_dict['train_full_sal_loss'] = train_full_sal_loss.item()
                    train_full_sal_loss = train_full.lambda_sal * train_full_sal_loss
                    train_full_sal_loss.backward() #at the begin use the image with label to pretrain the generater

                if train_full.lambda_adv:
                    if train_full_images is None or train_full_labels is None:
                        train_full_images = item['image'].to(self.device)
                        train_full_labels = item['label'].to(self.device)
                        train_full_labels.unsqueeze_(dim=1)
                    sal_pred_on_train_full_images = self.model.forward_G(train_full_images)
                    sal_pred_on_train_full_images_sigmoid = torch.sigmoid(sal_pred_on_train_full_images)
                    d_out,fo_d_out = self.model.forward_D(sal_pred_on_train_full_images_sigmoid)
                    fake_image_list.append(sal_pred_on_train_full_images_sigmoid.detach()) #the labeled image of the generator output
                    fake_with_gt_image = fake_image_list[-1]
                    train_full_adv_loss = self.adv_loss_function(d_out, self.real_label)#train the generater
                    # train_full_adv_loss += self.adv_loss_function(fo_d_out, self.real_label)
                    del fo_d_out
                    loss_dict['train_full_adv_loss'] = train_full_adv_loss.item()
                    train_full_adv_loss = train_full.lambda_adv * train_full_adv_loss
                    train_full_adv_loss.backward()

            # Training semi-supervised data
            train_semi_images = None
            if iter_time == train_semi.start_time+1:
                self.teacher_generator = ema(self.model.generator,0.99)
            if train_semi is not None:
                item = self.get_data(train_semi)
                if train_semi.lambda_sal and (train_semi.start_time is None or iter_time > train_semi.start_time):
                    train_semi_images = item['image'].to(self.device)
                    sal_pred_on_train_semi_images = self.model.forward_G(train_semi_images)#shengchengqi biaoqian
                    sal_pred_on_train_semi_images_sigmoid = torch.sigmoid(sal_pred_on_train_semi_images.detach())
                    d_out, fo_d_out = self.model.forward_D(sal_pred_on_train_semi_images_sigmoid)
                    sal_fake_label_on_train_semi_images = sal_pred_on_train_semi_images_sigmoid.gt_(0.5).float()#the wei biaoqian, binary the output of generator ,the mutiply the v 
                    weight = torch.sigmoid_(fo_d_out)
                    if train_semi.lambda_adv:
                        ignore_index = weight < train_semi.mask_T
                        sal_fake_label_on_train_semi_images[ignore_index] = self.ignore_value
                        del ignore_index
                    train_semi_sal_loss = self.sal_loss_function(sal_pred_on_train_semi_images, \
                                                             sal_fake_label_on_train_semi_images)#train the generator with v
                    loss_dict['train_semi_sal_loss'] = train_semi_sal_loss.item()
                  #  train_semi_sal_loss = train_semi.lambda_sal *( train_semi_sal_loss)#+teacher_generator_loss)
                    #train_semi_sal_loss.backward(retain_graph=True)
                    self.teacher_generator.update(self.model.generator)
                    generator_teacher_consistency = sigmoid_rampup(iter_time ,self.max_iter_time )
                    teacher_generator_out = self.teacher_generator.ema(train_semi_images)
                    teacher_generator_out_sigmoid = torch.sigmoid(teacher_generator_out.detach())
                    teacher_out, teacher_d_out =  self.model.forward_D(sal_pred_on_train_semi_images_sigmoid)
                    sal_fake_label_on_teacher_generator_out = teacher_generator_out_sigmoid.gt_(0.5).float()#the wei biaoqian, binary the output of generator ,the mutiply the v 
                    weight = torch.sigmoid_(teacher_d_out)
                    if train_semi.lambda_adv:
                        ignore_index = weight < train_semi.mask_T
                        sal_fake_label_on_teacher_generator_out[ignore_index] = self.ignore_value
                        del ignore_index
                    teacher_generator_loss = generator_teacher_consistency*self.teacher_generator_loss(sal_pred_on_train_semi_images,sal_fake_label_on_teacher_generator_out.detach())/8.0  #batchsize
                    #teacher_generator_loss.backward()
                    train_semi_sal_loss = train_semi.lambda_sal *( train_semi_sal_loss+teacher_generator_loss)
                    train_semi_sal_loss.backward()
                    del sal_fake_label_on_train_semi_images
                    del sal_pred_on_train_semi_images_sigmoid
                    del sal_pred_on_train_semi_images
                    del d_out
                    del fo_d_out
                    del weight
                if train_semi.lambda_adv:
                    if train_semi_images is None:
                        train_semi_images = item['image'].to(self.device)
                    sal_pred_on_train_semi_images = self.model.forward_G(train_semi_images)
                    sal_fake_label_on_train_semi_images = torch.sigmoid(sal_pred_on_train_semi_images)
                    d_out,fo_d_out = self.model.forward_D(sal_fake_label_on_train_semi_images)
                    fake_image_list.append(sal_fake_label_on_train_semi_images.detach())#train generator with unlabelled image
                    train_semi_adv_loss = self.adv_loss_function(d_out, self.real_label)
                    # train_semi_adv_loss += self.adv_loss_function(fo_d_out, self.real_label)
                    del fo_d_out
                    loss_dict['train_semi_adv_loss'] = train_semi_adv_loss.item()
                    train_semi_adv_loss = train_semi.lambda_adv * train_semi_adv_loss
                    train_semi_adv_loss.backward()

            # train D
            fake_image = None
            if len(fake_image_list) == 0:
                pass
            elif len(fake_image_list) == 1:
                fake_image = fake_image_list[0]
            elif len(fake_image_list) == 2:
                fake_image = torch.cat(fake_image_list,dim=0)

            if fake_image is not None and self.train_gt_data is not None:
                self.model.discriminator.freeze_bp(False)
                if fake_with_gt_image is not None:
                    self.model.discriminator.freeze_encoder_bp(True)#only train the v generator in the discriminor
                    d_out_fake,fo_d_out_fake = self.model.forward_D(fake_image_list[0])
                    d_out_label = 1 - torch.abs(fake_with_gt_image - train_full_labels)#v label
                    d_out_loss_on_fake = self.adv_loss_function(fo_d_out_fake[:d_out_label.size(0)],d_out_label)# update the generator to train v
                    d_out_loss_on_fake.backward()
                    #_,student_out = self.model.forward_D(fake_image_list[-1])#unlabeled images out
                    
                   # _,teacher_out = self.teacher_discriminator.ema(fake_image_list[-1])
                   # teacher_consistency = sigmoid_rampup(iter_time ,self.max_iter_time)
                   # mean_teacher_loss = teacher_consistency *  self.teacher_discriminator_loss(student_out,teacher_out.detach())/8.0#mse loss
                  #  mean_teacher_loss = self.teacher_discriminator_loss(student_out,teacher_out)#mse loss
                   # mean_teacher_loss.backward()
                    self.model.discriminator.freeze_encoder_bp(False)
                d_out_fake,fo_d_out_fake = self.model.forward_D(fake_image)
                d_out_loss_on_fake = self.adv_loss_function(d_out_fake,self.fake_label)
                d_out_loss_on_fake.backward()
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
                #self.teacher_discriminator.update(self.model.discriminator)
            self.G_optim.step()
           # self.teacher_generator.update(self.model.generator)
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
                self.model_saver(self.model.generator, save_base_name='G')
                self.model_saver(self.model.discriminator, save_base_name='D')
                if iter_time > train_semi.start_time:
                    self.model_saver(self.teacher_generator.ema, save_base_name='teacher')
        if self.model_saver is not None:
            self.model_saver(self.model.generator, isFinal=True, save_base_name='G')
            self.model_saver(self.model.discriminator, isFinal=True, save_base_name='D')
            self.model_saver(self.teacher_generator.ema, isFinal=True,save_base_name='teacher')
        if self.visual_helper is not None:
            self.visual_helper.close()
