from __future__ import print_function
from six.moves import range

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import os
import time
from PIL import Image, ImageFont, ImageDraw
from copy import deepcopy

from miscc.config import cfg
from miscc.utils import mkdir_p

from tensorboard import summary
from tensorboard import FileWriter
import tensorflow as tf

import model
from model import G_NET, D_NET64, D_NET128, D_NET256, D_NET512, D_NET1024, INCEPTION_V3



# ################## Shared functions ###################
def compute_mean_covariance(img):
    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width

    # batch_size * channel_num * 1 * 1
    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)

    # batch_size * channel_num * num_pixels
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels)
    # batch_size * num_pixels * channel_num
    img_hat_transpose = img_hat.transpose(1, 2)
    # batch_size * channel_num * channel_num
    covariance = torch.bmm(img_hat, img_hat_transpose)
    covariance = covariance / num_pixels

    return mu, covariance

def KL_loss(mu, logvar):  ## average kld on each dim
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def ce_loss(logq, p, average=True):
    N= 1
    if average:
        # N= p.shape[0]* p.shape[1] ## !!wrong
        N= p.shape[0]
    return -torch.sum(p* logq)/N

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)
def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
def compute_inception_score(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
            (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)
def negative_log_posterior_probability(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)


def load_network(gpus):  ## build, parallel, load networks
    print("+++Load/create network")
    shareGs= model.get_shareGs(cfg.GAN.GF_DIM)

    netG = G_NET(shareGs)
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    print(netG)

    entity_netG= model.COND_G_NET(cfg.GAN.ENTITY_DIM+ 1+ cfg.TEXT.DIMENSION, shareGs)
    entity_netG.apply(weights_init)
    entity_netG = torch.nn.DataParallel(entity_netG, device_ids=gpus)
    print(entity_netG)

    netsD = []
    # entity_netsD= []
    if cfg.TREE.BRANCH_NUM > 0:
        # netsD.append(D_NET64())
        # entity_netsD.append(model.ENTITY_D_NET64())
        ## use joint D --- 2.27.2019
        netsD.append(model.JOINT_D_NET64())
    if cfg.TREE.BRANCH_NUM > 1:        
        if cfg.TREE.SCALE==2:
            netsD.append(model.JOINT_D_NET128())            
        else:
            netsD.append(model.JOINT_D_NET256())  ## x4 scale            
    if cfg.TREE.BRANCH_NUM > 2:
        assert False, 'br3 todo'
        # netsD.append(D_NET256())
        # entity_netsD.append(model.ENTITY_D_NET256())
    
    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)
        # print(netsD[i])
        # entity_netsD[i].apply(weights_init)
        # entity_netsD[i] = torch.nn.DataParallel(entity_netsD[i], device_ids=gpus)
    print('Num of netsD', len(netsD))
    # print('Num of entity_netsD', len(entity_netsD))

    count = 0
    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        count = cfg.TRAIN.NET_G[istart:iend]
        count = int(count) + 1

    if cfg.TRAIN.NET_D != '':
        for i in range(len(netsD)):
            print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
            state_dict = torch.load('%s%d.pth' % (cfg.TRAIN.NET_D, i))
            netsD[i].load_state_dict(state_dict)    

    if cfg.CUDA:
        # netG.cuda()
        entity_netG.cuda()
        for i in range(len(netsD)):
            netsD[i].cuda()
            # entity_netsD[i].cuda()
    #     inception_model = inception_model.cuda()
    # inception_model.eval()  ## set eval mode

    # return netG, entity_netG, shareGs, netsD, len(netsD), count
    return entity_netG, shareGs, netsD, len(netsD), count


def define_optimizers(netG, netsD=[]):
    optimizersD = []
    num_Ds = len(netsD)
    for i in range(num_Ds):
        opt = optim.Adam(netsD[i].parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        optimizersD.append(opt)

    # G_opt_paras = []
    # for p in netG.parameters():
    #     if p.requires_grad:
    #         G_opt_paras.append(p)
    optimizerG = optim.Adam(netG.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999))
    return optimizerG, optimizersD


def save_model(netG, avg_param_G, netsD, epoch, model_dir):
    load_params(netG, avg_param_G)  ## save avg params, change G??
    torch.save(
        netG.state_dict(),
        '%s/netG_%d.pth' % (model_dir, epoch))
    for i in range(len(netsD)):
        netD = netsD[i]
        torch.save(
            netD.state_dict(),
            '%s/netD%d.pth' % (model_dir, i))
    print('Save G/Ds models.')


def save_img_results(imgs_tcpu, fake_imgs, num_imgs,
                     count, image_dir, summary_writer):
    num = cfg.TRAIN.VIS_COUNT

    # The range of real_img (i.e., self.imgs_tcpu[i][0:num])
    # is changed to [0, 1] by function vutils.save_image
    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(
        real_img, '%s/real_samples.png' % (image_dir),
        normalize=True)
    real_img_set = vutils.make_grid(real_img).numpy()
    real_img_set = np.transpose(real_img_set, (1, 2, 0))
    real_img_set = real_img_set * 255
    real_img_set = real_img_set.astype(np.uint8)
    # sup_real_img = summary.image('real_img', real_img_set)
    # summary_writer.add_summary(sup_real_img, count)

    for i in range(num_imgs):  ## stages
        fake_img = fake_imgs[i][0:num]
        # The range of fake_img.data (i.e., self.fake_imgs[i][0:num])
        # is still [-1. 1]...
        vutils.save_image(
            fake_img.data, '%s/epoch_%03d_fake_samples%d.png' %
            (image_dir, count, i), normalize=True)

        fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()

        fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
        fake_img_set = (fake_img_set + 1) * 255 / 2
        fake_img_set = fake_img_set.astype(np.uint8)

        # sup_fake_img = summary.image('fake_img%d' % i, fake_img_set)
        # summary_writer.add_summary(sup_fake_img, count)
        # summary_writer.flush()


# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, imsize):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])  ## as default
        #torch.cuda.device(self.gpus[0])  ## as default
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)  ## batches number, drop last

    def prepare_data(self, data):
        imgs, w_imgs, t_embedding, cls, _ = data
        #print(cls, cls.shape)
        #assert torch.max(cls)<200
        # print(imgs)
        # print(cls[0], cls.shape)
        # cls= cls.long()
        # cls-=1  ## from 0 to 199

        real_vimgs, wrong_vimgs = [], []
        if cfg.CUDA:
            vembedding = Variable(t_embedding).cuda()
            vcls= Variable(cls).cuda()
        else:
            vembedding = Variable(t_embedding)
            vcls= Variable(cls)
        for i in range(self.num_Ds):
            if cfg.CUDA:
                real_vimgs.append(Variable(imgs[i]).cuda())  ## stages
                wrong_vimgs.append(Variable(w_imgs[i]).cuda())
            else:
                real_vimgs.append(Variable(imgs[i]))
                wrong_vimgs.append(Variable(w_imgs[i]))
        return imgs, real_vimgs, wrong_vimgs, vembedding, vcls  ## imgs for save real images

    def onehot(self, cls_vec, n):
        bs= cls_vec.shape[0]
        one_hot= torch.zeros(bs, n)
        for i in range(bs):
            one_hot[i, cls_vec[i]]= 1
        if cfg.CUDA:
            one_hot= Variable(one_hot).cuda()
        else:
            one_hot= Variable(one_hot)
        return one_hot

    def multihot(self, classes_vec, n):
        ## class num from 1; classes_vec is 2d list, cols are different
        # bs= classes_vec.shape[0]
        bs= self.batch_size
        print(bs, classes_vec)
        multi_hot= torch.zeros(bs, n)
        for i in range(bs):
            print(i)
            if not classes_vec[i]:
                multi_hot[i, n-1]= 1
                continue
            for j in classes_vec[i]:
                multi_hot[i, j-1]= 1
        if cfg.CUDA:
            multi_hot= Variable(multi_hot).cuda()
        else:
            multi_hot= Variable(multi_hot)
        return multi_hot

    def train_entity_Dnet(self, idx, count):
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)        

        entity_netD, optD = self.entity_netsD[idx], self.entity_optsD[idx]
        real_imgs = self.real_imgs[idx]        
        fake_imgs = self.fake_imgs[idx]
        
        entity_netD.zero_grad()
        # Forward
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]        
        real_tp, real_cp = entity_netD(real_imgs)        
        fake_tp, fake_cp = entity_netD(fake_imgs.detach())
        # for reality
        errD_real = self.bce_logit(real_tp, real_labels)+ self.bce_logit(fake_tp, fake_labels)
        # for entity class
        errD_class = self.nll(real_cp, self.cls)+ self.nll(fake_cp, torch.ones(batch_size).long().cuda()* cfg.GAN.ENTITY_DIM)
        errD= errD_real+ errD_class        
        # backward
        errD.backward()
        # update parameters
        optD.step()
        # log
        if flag == 0:
            summary_D = summary.scalar('entity_D_loss%d' % idx, errD.data[0])
            self.summary_writer.add_summary(summary_D, count)
        return errD

    def loss_entity_Gnet(self, count):
        #self.entity_optG.zero_grad()        
        flag = count % 100  ## log each iter 100
        batch_size = self.real_imgs[0].size(0)
        mu, logvar = self.entity_mu, self.entity_logvar  ## entity ca
        real_labels = self.real_labels[:batch_size]
        errG_total = 0

        for i in range(self.num_Ds):
            tp, cp = self.entity_netsD[i](self.fake_imgs[i])
            errG = self.bce_logit(tp, real_labels)+ self.nll(cp, self.cls)            
            errG_total = errG_total + errG  # add all stage generators losses
            if flag == 0:
                summary_D = summary.scalar('G_loss%d' % i, errG.data[0])
                self.summary_writer.add_summary(summary_D, count)

        kl_loss = KL_loss(mu, logvar) * cfg.TRAIN.COEFF.KL
        errG_total = errG_total + kl_loss ## add kl
        #errG_total.backward(retain_graph= True)  ## *update together
        # errG_total.backward()
        # self.entity_optG.step()
        return kl_loss, errG_total

    def train_joint_Dnet(self, idx, count):
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)
        criterion, mu = self.criterion, self.mu  # mean text embedding

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        real_imgs = self.real_imgs[idx]
        wrong_imgs = self.wrong_imgs[idx]  # mismatch imgs
        fake_imgs = self.fake_imgs[idx]
        #
        netD.zero_grad()
        # Forward
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]        
        real_logits = netD(real_imgs, mu.detach())
        wrong_logits = netD(wrong_imgs, mu.detach())  ## mismatch
        fake_logits = netD(fake_imgs.detach(), mu.detach())
        
        # for matching, real or not for pair data
        errD_real = criterion(real_logits[0], real_labels)
        errD_wrong = criterion(wrong_logits[0], fake_labels)
        errD_fake = criterion(fake_logits[0], fake_labels)
        
        # for reality
        if len(real_logits) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:  ## uncond coeff 1.0
            errD_real_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * criterion(real_logits[1], real_labels)
            errD_wrong_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * criterion(wrong_logits[1], real_labels)
            errD_fake_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * criterion(fake_logits[1], fake_labels)
            #
            # errD_real = errD_real + errD_real_uncond
            # errD_wrong = errD_wrong + errD_wrong_uncond  ##? double real input for uncond
            # errD_fake = errD_fake + errD_fake_uncond
            errD_match = errD_real+ errD_wrong+ errD_fake
            errD_uncond = errD_real_uncond+ errD_wrong_uncond+ errD_fake_uncond            
            # errD_cls = self.nll(real_logits[2], self.cls)+ \
                # self.nll(fake_logits[2], torch.ones(batch_size).long().cuda()* cfg.GAN.ENTITY_DIM)
            
            # print(self.cls_label[0], self.real_cp[0])
            fake_cp= torch.zeros(batch_size, cfg.GAN.ENTITY_DIM+1); fake_cp[:,-1]= 1
            # print(fake_cp[0])
            errD_cls= self.CE(real_logits[2], self.real_cp)+ \
                self.CE(fake_logits[2], fake_cp.cuda())

            # errD = errD_real + errD_wrong + errD_fake + errD_cls  ## 3+ vs 3- also balanced?
            errD= errD_match+ errD_uncond+ errD_cls
        else:
            errD = errD_real + 0.5 * (errD_wrong + errD_fake)  ## balance +/- samples
        
        # backward
        errD.backward()
        # update parameters
        optD.step()

        # log
        if flag == 0:
            summary_D = summary.scalar('D_loss%d' % idx, errD.data[0])
            self.summary_writer.add_summary(summary_D, count)
        return errD, errD_match, errD_uncond, errD_cls

    def loss_joint_Gnet(self, count):
        #self.netG.zero_grad()        
        flag = count % 100  ## log each iter 100
        batch_size = self.real_imgs[0].size(0)
        criterion, mu, logvar = self.criterion, self.mu, self.logvar
        real_labels = self.real_labels[:batch_size]

        errGs_match=0; errGs_uncond= 0; errGs_cls= 0        
        for i in range(self.num_Ds):
            outputs = self.netsD[i](self.fake_imgs[i], mu)  ## ca net also tuned
            errGs_match+= criterion(outputs[0], real_labels)  # matching loss
            
            if len(outputs) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
                errGs_uncond+= cfg.TRAIN.COEFF.UNCOND_LOSS*criterion(outputs[1], real_labels)  ## reality loss
                errGs_cls+= self.CE(outputs[2], self.real_cp)                            
            # if flag == 0:
            #     summary_D = summary.scalar('G_loss%d' % i, errG.data[0])
            #     self.summary_writer.add_summary(summary_D, count)        
        errG_total = errGs_match+ errGs_uncond+ errGs_cls  # add all stage generators losses
        
        kl_loss = KL_loss(mu, logvar)
        # entity_kl_loss = KL_loss(self.entity_mu, self.entity_logvar)
        errG_total = errG_total + kl_loss * cfg.TRAIN.COEFF.KL  ## add kl        

        return errG_total, errGs_match, errGs_uncond, errGs_cls, kl_loss


    def train(self):        
        self.netG, self.shareGs, self.netsD, \
            self.num_Ds, start_count = load_network(self.gpus);
        ## Memory is not enough, use trade-off eval when training.
        if cfg.TRAIN.BIG_EVAL == False:
            self.inception_model = INCEPTION_V3().cuda()
            self.inception_model.eval()
        # avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD = define_optimizers(self.netG, self.netsD)
        # self.entity_optG, _= define_optimizers(self.entity_netG, self.netsD)

        self.criterion = nn.BCELoss()
        self.bce_logit= nn.BCEWithLogitsLoss() ## default average loss, logit->sigmoid
        self.nll= nn.NLLLoss()
        # self.CE= nn.CrossEntropyLoss() ##!! in fact, NLL loss, only support one-hot target
        self.CE= ce_loss

        self.real_labels = Variable(torch.FloatTensor(self.batch_size).fill_(1))
        self.fake_labels = Variable(torch.FloatTensor(self.batch_size).fill_(0))        

        self.gradient_one = torch.FloatTensor([1.0])
        self.gradient_half = torch.FloatTensor([0.5])

        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(self.batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))  ## same for each iter

        if cfg.CUDA:
            self.criterion.cuda()
            self.nll.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()
            self.gradient_one = self.gradient_one.cuda()
            self.gradient_half = self.gradient_half.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        predictions = []
        count = start_count  # iter num
        start_epoch = start_count // (self.num_batches)
        print("Num_batches: %d"%self.num_batches)
        print('+++Start training...')
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            for step, data in enumerate(self.data_loader, 0):
                if step<3: print('Iter %d'%step)
                #######################################################
                # (0) Prepare training data  ## be Variable
                ######################################################
                self.imgs_tcpu, self.real_imgs, self.wrong_imgs, self.txt_embedding, \
                    self.cls_label = self.prepare_data(data);  ## true img on cpu
                # self.cls_label= self.multihot(self.cls, cfg.GAN.ENTITY_DIM+ 1) ## 1 for misc class/no label
                # self.cls_label= Variable(self.cls).cuda()
                self.real_cp= self.cls_label/(torch.sum(self.cls_label,1).view(-1, 1))
                # print(self.real_cp[0])

                #######################################################
                # (1) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)  ## share noise for two stgs
                self.cond_info= torch.cat((self.txt_embedding, self.cls_label), 1)
                self.hcodes, self.mu, self.logvar, self.std= \
                    self.netG(noise, self.cond_info);
                self.fake_imgs= self.netG.module.image(self.hcodes)                

                #######################################################
                # (2) Update D network
                ######################################################
                errDs = 0; errDs_match=0; errDs_uncond=0; errDs_cls=0
                # entity_errD_allstgs = 0
                for i in range(self.num_Ds):
                    errD, errD_match, errD_uncond, errD_cls= self.train_joint_Dnet(i, count)
                    errDs+= errD; errDs_match+=errD_match
                    errDs_uncond+= errD_uncond; errDs_cls+= errD_cls
                    # entity_errD= self.train_entity_Dnet(i, count)
                    # entity_errD_allstgs+= entity_errD

                # #######################################################
                # # (3) Update G network: maximize log(D(G(z)))
                # ######################################################
                self.netG.zero_grad()                
                # entity_kl_loss, entity_errG_allstgs = self.loss_entity_Gnet(count)
                errGs, errGs_match, errGs_uncond, \
                    errGs_cls, kl_loss = self.loss_joint_Gnet(count)                
                errGs.backward()  
                # self.entity_optG.step() 
                # self.netG.module.img_net1.zero_grad()
                # self.netG.module.img_net2.zero_grad()  ##!! not clear zero before 2.22
                self.optimizerG.step()                 
                # # for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                # #     avg_p.mul_(0.999).add_(0.001, p.data)


                ## for inception score  ## acorss many iters
                if cfg.TRAIN.BIG_EVAL == False and step%2:
                    # print(self.fake_imgs[-1].shape)
                    pred = self.inception_model(self.fake_imgs[-1].detach())
                    predictions.append(pred.data.cpu().numpy())                

                count = count + 1

                ## snapshot
                # if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  ## iter 1k
                    # save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
                    ## Save images
                    # backup_para = copy_G_params(self.netG)
                    # load_params(self.netG, avg_param_G)
                    # # use avg_param gen imgs
                    # # self.fake_imgs, _, _ = self.netG(fixed_noise, self.txt_embedding)
                    # save_img_results(self.imgs_tcpu, self.fake_imgs, self.num_Ds,
                    #                  count, self.image_dir, self.summary_writer)
                    # # load backup params
                    # load_params(self.netG, backup_para)                    

            ## a epoch ends
            end_t = time.time()  ## for one epoch
            print('''[%d/%d][BN=%d][%d stages]
                Loss_D_all: %.2f Loss_D_match: %.2f Loss_D_uncond: %.2f Loss_D_cls: %.2f 
                Loss_G_all: %.2f Loss_G_match: %.2f Loss_G_uncond: %.2f Loss_G_cls: %.2f 
                Loss_KL: %.2f
                Time: %.2fs
                '''  # D(real): %.4f D(wrong):%.4f  D(fake) %.4f
                % (epoch, self.max_epoch, self.num_batches, self.num_Ds,
                    errDs.data[0], errDs_match.data[0], errDs_uncond.data[0], errDs_cls.data[0],
                    errGs.data[0], errGs_match.data[0], errGs_uncond.data[0], errGs_cls.data[0],
                    kl_loss.data[0],
                    end_t - start_t))
            print()
            # print('Entity mu and std (mean)\n', torch.mean(self.entity_mu,0), '\n', torch.mean(self.entity_std,0))
            print('Sentence and entity mu and std (mean)\n', torch.mean(self.mu,0), '\n', torch.mean(torch.exp(0.5 * self.logvar),0))
            # print('1-5 rows of entity mu, std')
            # print(self.entity_mu[0:5,:], '\n', self.entity_std[0:5,:])
            # print('1-5 rows of sentence mu, std')
            # print(self.mu[0:5,:], '\n', torch.exp(0.5 * self.logvar)[0:5,:])
            print()
            save_img_results(self.imgs_tcpu, self.fake_imgs, self.num_Ds,
                                     epoch, self.image_dir, self.summary_writer)
            print('Save images ok')
            ## snapshot
            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == cfg.TRAIN.SNAPSHOT_INTERVAL-1:  ## /50 epochs               
               torch.save(self.netG.state_dict(), '%s/netG_epoch%d.pth'%(self.model_dir, epoch))
               # torch.save(self.entity_netG.state_dict(), '%s/entity_netG_epoch%d.pth'%(self.model_dir, epoch))
               print('Snapshot: save model ok')
            
            ## real-time eval
            THR= 6 ## FOR COCO
            SPLIT_N= 10
            if cfg.TRAIN.BIG_EVAL==False: 
                print('\n+++Use eval across iters...')               
                if len(predictions)* self.batch_size >= 3000:  
                    predictions = np.concatenate(predictions, 0)
                    print('Eval images num: %d'%predictions.shape[0])
                    mean, std = compute_inception_score(predictions, SPLIT_N)
                    print('Epoch%d IS: %.3f +- %.3f'%(epoch, mean, std))
                    # print('mean:', mean, 'std', std)
                    # m_incep = summary.scalar('Inception_mean', mean)
                    # self.summary_writer.add_summary(m_incep, count)
                    # #
                    # mean_nlpp, std_nlpp = \
                    #     negative_log_posterior_probability(predictions, 10)  ## cal be bird prob
                    # m_nlpp = summary.scalar('NLPP_mean', mean_nlpp)
                    # self.summary_writer.add_summary(m_nlpp, count)
                    # #
                    predictions = []
                    if mean>THR:                          
                        torch.save(self.netG.state_dict(), '%s/netG_epoch%d.pth'%(self.model_dir, epoch))
                        # torch.save(self.entity_netG.state_dict(), '%s/entity_netG_epoch%d.pth'%(self.model_dir, epoch))
                        print('Saved good model')
                continue
            
            ## if memory enough   
            print('Use eval on each epoch.') 
            thr= -1
            #thr= 99
            if epoch>=thr:# and epoch%2:
                n= 1000
                fake_imgs= []
                for step, data in enumerate(self.data_loader, 0):
                    #print(step)
                    if len(fake_imgs) * cfg.TRAIN.BATCH_SIZE> n:
                        break  ## next enumerate will repackage dataloader and bn not be less
                    _, _, _, self.txt_embedding, self.cls = self.prepare_data(data)
                    self.cls_onehot= self.onehot(self.cls, cfg.GAN.ENTITY_DIM)
                    noise.data.normal_(0, 1)
                    hcodes, _, _ = self.netG(noise, self.txt_embedding)  ##!! don't use self.sen_hcodes, will not release mem after for
                    # entity_hcodes, _, _, _ = self.entity_netG(noise, self.cls_onehot);                    
                    _fake_imgs= self.netG.module.image(hcodes)
                    ##!!! must detach, else exceed mem
                    fake_imgs.append(_fake_imgs[-1].detach().cpu())
                fake_imgs= torch.cat(fake_imgs, 0)
                print(fake_imgs.shape)
                assert fake_imgs.shape[0]>n, '%d'%fake_imgs.shape[0]
                fake_imgs= ((fake_imgs.cpu()+1)*(255.99/2)).numpy().astype('int32').transpose(0,2,3,1)
                ## cal is
                import inception_score
                IS= inception_score.get_inception_score(list(fake_imgs), 1)
                print('Epoch%d IS: %.3f\n'%(epoch, IS[0]))
                if IS[0]>5.4:                          
                    torch.save(self.netG.state_dict(), '%s/netG_epoch%d.pth'%(self.model_dir, epoch))
                    # torch.save(self.entity_netG.state_dict(), '%s/entity_netG_epoch%d.pth'%(self.model_dir, epoch))
                    print('Saved good model')
        
        self.summary_writer.close()

    def save_superimages(self, images_list, filenames,
                         save_dir, split_dir, imsize):
        batch_size = images_list[0].size(0)
        num_sentences = len(images_list)
        for i in range(batch_size):
            s_tmp = '%s/super/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            #
            savename = '%s_%d.png' % (s_tmp, imsize)
            super_img = []
            for j in range(num_sentences):
                img = images_list[j][i]
                # print(img.size())
                img = img.view(1, 3, imsize, imsize)
                # print(img.size())
                super_img.append(img)
                # break
            super_img = torch.cat(super_img, 0)
            vutils.save_image(super_img, savename, nrow=10, normalize=True)

    def save_singleimages(self, images, filenames,
                          save_dir, split_dir, sentenceID, cls, imsize, noiseID):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s' %\
                (save_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            # fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
            fullpath = '%s_%d_class%d_sid%d_nid%d.png' % (s_tmp, imsize, cls[i], sentenceID, noiseID)
            # range from [-1, 1] to [0, 255]
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def evaluate(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            # if split_dir == 'test':
            #     split_dir = 'valid'
            shareGs= model.get_shareGs(cfg.GAN.GF_DIM)

            netG = G_NET(shareGs)
            netG.apply(weights_init)
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            print(netG)

            entity_netG= model.COND_G_NET(cfg.GAN.ENTITY_DIM, shareGs)
            entity_netG.apply(weights_init)
            entity_netG = torch.nn.DataParallel(entity_netG, device_ids=self.gpus)
            print(entity_netG)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load ', cfg.TRAIN.NET_G)

            state_dict = \
                torch.load(cfg.TRAIN.ENTITY_NET_G, map_location=lambda storage, loc: storage)
            entity_netG.load_state_dict(state_dict)
            print('Load ', cfg.TRAIN.ENTITY_NET_G)
    

            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(self.batch_size, nz))
            if cfg.CUDA:
                netG.cuda()
                entity_netG.cuda()
                noise = noise.cuda()            
            
            ## switch to evaluate mode        
            # for i in range(1):
            #   print('epoch'+str(i))
            #   for step, data in enumerate(self.data_loader, 0): 
            #       print(step)
            #       #if step > 50: break
            #       _, t_embeddings, cls, filenames = data
            #       cls-=1
            #       cls_onehot= self.onehot(cls, cfg.GAN.ENTITY_DIM)
            #       if cfg.CUDA:
            #           t_embeddings = Variable(t_embeddings).cuda()
            #           cls_onehot = Variable(cls_onehot).cuda()
            #       else:
            #           t_embeddings = Variable(t_embeddings)
            #           cls_onehot= Variable(cls_onehot)

            #       noise.data.normal_(0, 1)  ## variable in for
            #       entity_hcodes, _, _, _ = entity_netG(noise, cls_onehot)
            #       sen_hcodes, _, _ = netG(noise, t_embeddings[:, 0, :])
            
            eval_mode= False
            if eval_mode:
                netG.eval(); entity_netG.eval()
                mode= 'evalmode'
            else:
                mode= 'trainmode'
            evalset= 'Testset'
            embedding_dim= 10
            # one_cap= True; embedding_dim= 20
            sample_num= 3 ## one cap n noise

            if embedding_dim== 1 and sample_num>1:
                save_dir= '%s_%s_onecap_varnoise_%s_%s' % (evalset, mode, cfg.TRAIN.NET_G[:-4].split('_')[-1], cfg.TRAIN.NET_G.split('/')[-3])
            elif embedding_dim>1 and sample_num>1:
                save_dir= '%s_%s_varnoise_%s_%s' % (evalset, mode, cfg.TRAIN.NET_G[:-4].split('_')[-1], cfg.TRAIN.NET_G.split('/')[-3])
            else:
                save_dir= '%s_%s_fixednoise_%s_%s' % (evalset, mode, cfg.TRAIN.NET_G[:-4].split('_')[-1], cfg.TRAIN.NET_G.split('/')[-3])                            
            print('Save to %s'%save_dir)
            

            for step, data in enumerate(self.data_loader, 0):
                # if step >6: break

                imgs, t_embeddings, cls, filenames = data
                cls-=1
                cls_onehot= self.onehot(cls, cfg.GAN.ENTITY_DIM)
                if cfg.CUDA:
                    t_embeddings = Variable(t_embeddings).cuda()
                    cls_onehot = Variable(cls_onehot).cuda()
                else:
                    t_embeddings = Variable(t_embeddings)
                    cls_onehot= Variable(cls_onehot)
                # print(t_embeddings[:, 0, :], t_embeddings.size(1))
                # print(filenames) ## include class folder name 'aaa/bbb'

                # embedding_dim = t_embeddings.size(1)  ## one image n sentences                
                # embedding_dim= 1 ## for one-vs-many images
                batch_size = imgs[0].size(0)
                noise.data.resize_(batch_size, nz)
                # noise.data.normal_(0, 1) ## minor var in captions                


                fake_img_list = []
                for i in range(embedding_dim):                    
                    for j in range(sample_num):
                        noise.data.normal_(0, 1)  ## for more various
                        sen_hcodes, _, _ = netG(noise, t_embeddings[:, i, :])  ##!! don't use self.sen_hcodes, will not release mem after for                        
                        #fake_imgs, _, _ = netG(noise, t_embeddings[:, i, :])                                     
                        entity_hcodes, _, _, _ = entity_netG(noise, cls_onehot);                    
                        fake_imgs= netG.module.get_image(entity_hcodes, sen_hcodes)
                        if cfg.TEST.B_EXAMPLE:
                            # fake_img_list.append(fake_imgs[0].data.cpu())
                            # fake_img_list.append(fake_imgs[1].data.cpu())
                            fake_img_list.append(fake_imgs[-1].data.cpu())
                        else:
                            self.save_singleimages(fake_imgs[-1].data.cpu(), filenames,
                                               save_dir, split_dir, i, cls, 128, j)
                            # self.save_singleimages(fake_imgs[-2], filenames,
                            #                        save_dir, split_dir, i, 128)
                            # self.save_singleimages(fake_imgs[-3], filenames,
                            #                        save_dir, split_dir, i, 64)
                        # break
                
                if cfg.TEST.B_EXAMPLE:
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 64)
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 128)
                    self.save_superimages(fake_img_list, filenames,
                                          save_dir, split_dir, 256)
                print('[%d/%d]'% (step, self.num_batches))
            print('Save images ok')