import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from scipy.stats import truncnorm as tn
import numpy as np

from capsule_layer import modules as capsule


# ############################## For Compute inception score ##############################
# Besides the inception score computed by pretrained model, especially for fine-grained datasets (such as birds, bedroom),
#  it is also good to compute inception score using fine-tuned model and manually examine the image quality.
class INCEPTION_V3(nn.Module):
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()  ## trochvision has
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # print(next(model.parameters()).data)
        state_dict = \
            model_zoo.load_url(url, model_dir='../data', map_location=lambda storage, loc: storage)  ## download params
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        print('Loaded pretrained model')
        # print(next(self.model.parameters()).data)
        # print(self.model)

    def forward(self, input):
        # [-1.0, 1.0] --> [0, 1.0]
        x = input * 0.5 + 0.5
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]  ## ??which dataset
        # --> mean = 0, std = 1
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        #
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)  ## no recover image to 255
        # 299 x 299 x 3
        x = self.model(x)
        x = nn.Softmax()(x)
        return x


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)  ## Call .contiguous() before .view() if permute


class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.shape= args

    def forward(self, x):
        return x.permute(self.shape).contiguous()


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):  # channels no change
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )


    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self, cond_dim= None):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        if cond_dim is not None:
            self.fc = nn.Linear(cond_dim, self.ef_dim * 4, bias=True)
        else:
            self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu), std

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code, std = self.reparametrize(mu, logvar)
        return c_code, mu, logvar, std


class VC_NET(nn.Module):
    '''VC_NET for inferring conditional manifold to augment condition'''
    def __init__(self, cond_dim):
        super(VC_NET,self).__init__()
        self.cond_dim = cond_dim
        self.noise_dim = cfg.GAN.Z_DIM
        self.manifd_dim = cfg.GAN.MANIFD_DIM
        self.threshold= -1 ## truncate dist when test        

        self.fc1 = nn.Linear(self.cond_dim+ self.noise_dim, 512) ## to get attention
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2= nn.BatchNorm1d(256)
        self.fc31 = nn.Linear(256, self.manifd_dim) # mu
        self.fc32 = nn.Linear(256, self.manifd_dim) # log(variance) , attention

    def encode(self, x):
        h = F.relu(self.bn_fc1(self.fc1(x)))
        h = F.relu(self.bn_fc2(self.fc2(h)))  # use bn in encoder
        # h = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc31(h), self.fc32(h)

    def reparameterize(self, mu, logvar, seed):    
        std = torch.exp(0.5 * logvar)    
        return seed.mul(std).add_(mu), std

    def forward(self, noise, cond):        
        x= torch.cat((noise, cond), 1)
        self.bs= x.shape[0]

        mu, logvar = self.encode(x)
        if self.training:  ## add when thinking cal metric
            seed= torch.randn(self.bs, self.manifd_dim)
        else:
            if self.threshold>0:
                seed= torch.tensor(tn.rvs(-self.threshold, self.threshold, size= self.bs* self.manifd_dim), dtype= torch.float).view(self.bs, self.manifd_dim)
            else:
                seed= torch.randn(self.bs, self.manifd_dim)
        if cfg.CUDA: seed= seed.cuda()

        c, std = self.reparameterize(mu, logvar, seed)
        return c, mu, logvar, std


class COND_INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super(COND_INIT_STAGE_G, self).__init__()
        if cfg.TRAIN.CAT_Z == 'concat':
            self.in_dim = cfg.GAN.MANIFD_DIM * 2
        else:
            self.in_dim = cfg.GAN.MANIFD_DIM
        self.gf_dim= ngf

        self.fc= nn.Sequential(
            nn.Linear(self.in_dim, ngf * 4 * 4 * 2, bias=False),  # *2 FOR GLU
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, ac_x):
        # state size 16ngf x 4 x 4
        out_code = self.fc(ac_x)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size 8ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 4ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size 2ngf x 32 x 32
        out_code = self.upsample3(out_code)
        # state size ngf x 64 x 64
        out_code = self.upsample4(out_code)

        return out_code


class COND_INIT_STAGE_G_withCap(nn.Module):
    def __init__(self, ngf):
        super(COND_INIT_STAGE_G_withCap, self).__init__()
        self.in_dim = cfg.GAN.MANIFD_DIM
        self.gf_dim= ngf ## 1024
        self.bs= cfg.TRAIN.BATCH_SIZE     

        self.fc_cap= nn.Sequential(
            ## capsule: 32*8=> 1024*16 !ERROR - !Good
            Reshape(self.bs, -1, 8),  ## make it 3d
            capsule.CapsuleLinear(out_capsules= ngf, in_length=8, out_length= 4*4* 2, in_capsules= None),
            # capsule.CapsuleLinear(out_capsules= 4*4, in_length=8, out_length= ngf, in_capsules= 16, share_weight=False),
            # Permute(0, 2, 1), ## [B, 16, ngfx2] -> [B, ngfx2, 16]
            Reshape(-1, ngf*4*4* 2),
            nn.BatchNorm1d(ngf*4*4* 2),
            GLU()
            )

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z, noise=None):
        if noise is not None:  ## != error!!!
            # print('intooooooooooooooooooo')
            z= torch.cat((z, noise), 1)  ## for CA
        out_code = self.fc_cap(z)
        # state size 16ngf x 4 x 4
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size 8ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 4ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size 2ngf x 32 x 32
        out_code = self.upsample3(out_code)
        # state size ngf x 64 x 64
        out_code = self.upsample4(out_code)

        return out_code


class COND_INIT_STAGE_G_Exchange_Cap(nn.Module):
    def __init__(self, ngf):
        super(COND_INIT_STAGE_G_Exchange_Cap, self).__init__()
        self.in_dim = cfg.GAN.MANIFD_DIM
        self.gf_dim= ngf ## 1024
        self.bs= cfg.TRAIN.BATCH_SIZE     

        self.fc_cap= nn.Sequential(
            ## capsule: 16*8=> 1024*16 !ERROR - !Good
            Reshape(self.bs, -1, 8),  ## make it 3d
            capsule.CapsuleLinear(out_capsules= (ngf//2)*2, in_length=8, out_length= 4*4, in_capsules= None), ## 7/23, glu on ngf
            # capsule.CapsuleLinear(out_capsules= ngf, in_length=8, out_length= 4*4* 2, in_capsules= None),  ## weird glu on out_length, 2019/7/23
            # capsule.CapsuleLinear(out_capsules= 4*4, in_length=8, out_length= ngf, in_capsules= 16, share_weight=False),
            # Permute(0, 2, 1), ## [B, 16, ngfx2] -> [B, ngfx2, 16]
            Reshape(-1, (ngf//2)*4*4* 2),
            nn.BatchNorm1d((ngf//2)*4*4* 2),
            GLU()
            )
        self.fc_cap1= nn.Sequential(
            ## capsule: 16*8=> 1024*16 !ERROR - !Good
            Reshape(self.bs, -1, 8),  ## make it 3d
            capsule.CapsuleLinear(out_capsules= (ngf//2)*2, in_length=8, out_length= 4*4, in_capsules= None), ## 7/23, glu on ngf
            # capsule.CapsuleLinear(out_capsules= ngf, in_length=8, out_length= 4*4* 2, in_capsules= None),  ## weird glu on out_length, 2019/7/23
            # capsule.CapsuleLinear(out_capsules= 4*4, in_length=8, out_length= ngf, in_capsules= 16, share_weight=False),
            # Permute(0, 2, 1), ## [B, 16, ngfx2] -> [B, ngfx2, 16]
            Reshape(-1, (ngf//2)*4*4* 2),
            nn.BatchNorm1d((ngf//2)*4*4* 2),
            GLU()
            )

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z):
        zs, zc= z[:, :self.in_dim], z[:, self.in_dim:]
        zs, zc= zs.contiguous(), zc.contiguous()
        zs_out_code = self.fc_cap(zs); zc_out_code = self.fc_cap1(zc) ## first capsule
        # zs_out_code, zc_out_code = zs_out_code.contiguous(), zc_out_code.contiguous()
        zs_out_code = zs_out_code.view(-1, self.gf_dim//2, 4, 4)
        zc_out_code = zc_out_code.view(-1, self.gf_dim//2, 4, 4)
        # 1024 x 4 x4
        out_code = torch.cat((zs_out_code, zc_out_code), 1)  ## then fuse
        # state size 8ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 4ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size 2ngf x 32 x 32
        out_code = self.upsample3(out_code)
        # state size ngf x 64 x 64
        out_code = self.upsample4(out_code)

        return out_code


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if cfg.GAN.B_CONDITION:
            self.in_dim = cfg.GAN.Z_DIM + cfg.GAN.EMBEDDING_DIM
        else:
            self.in_dim = cfg.GAN.Z_DIM
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),  # *2 FOR GLU
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)  ## 4x16， 64x64xngf

    def forward(self, z_code, c_code=None):
        if cfg.GAN.B_CONDITION and c_code is not None:
            in_code = torch.cat((c_code, z_code), 1)
        else:
            in_code = z_code
        # state size 16ngf x 4 x 4
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size 8ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 4ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size 2ngf x 32 x 32
        out_code = self.upsample3(out_code)
        # state size ngf x 64 x 64
        out_code = self.upsample4(out_code)

        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, num_residual=cfg.GAN.R_NUM):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if cfg.GAN.B_CONDITION:
            if cfg.TRAIN.CAT_Z== 'concat':
                self.ef_dim = cfg.GAN.EMBEDDING_DIM * 2  ### **** for concat z1, z2 ****
            else:
                self.ef_dim = cfg.GAN.EMBEDDING_DIM
        else:
            self.ef_dim = cfg.GAN.Z_DIM
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):  # stack resblocks
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim  # embedding channels

        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf // 2)
        if cfg.TREE.SCALE==4:
            self.upsample2 = upBlock(ngf//2, ngf//4)

    def forward(self, h_code, c_code):
        s_size = h_code.size(2)  # 64x64,,,  
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)  # replicate embedding
        # state size (ngf+egf) x in_size x in_size
        h_c_code = torch.cat((c_code, h_code), 1)  # cat along on depth
        # state size ngf x in_size x in_size
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)
        if cfg.TREE.SCALE==4:
            out_code = self.upsample2(out_code)  ## ngf/4 x 4in_size

        return out_code


class GET_IMAGE_G(nn.Module):  # decoder for hcode
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img

def get_shareGs(gf_dim):
    share_Gs=[]

    if cfg.TREE.BRANCH_NUM > 0:        
        img_net1 = GET_IMAGE_G(gf_dim)
        share_Gs.append(img_net1)
    if cfg.TREE.BRANCH_NUM > 1:        
        img_net2 = GET_IMAGE_G(gf_dim // cfg.TREE.SCALE)
        share_Gs.append(img_net2)
    if cfg.TREE.BRANCH_NUM > 2:        
        img_net3 = GET_IMAGE_G(gf_dim // cfg.TREE.SCALE**2)
        share_Gs.append(img_net3)
    return share_Gs


### Build generator by split two z and use CA INSTEAD OF VC for sentence inference
class COND_G_NET_CATZ_CA(nn.Module):  ## entity_gnet
    def __init__(self, sen_dim, cls_dim, share_Gs, use_cap= False, cat= 'concat', exchange= False):
        super(COND_G_NET_CATZ_CA, self).__init__()
        self.gf_dim= cfg.GAN.GF_DIM
        self.ca_net1= CA_NET()  ## INSTEAD OF VC IN SENTENCE inference
        # self.ca_net2= CA_NET(cls_dim)  ## FOR CLASS INFERENCE
        self.vc_net2= VC_NET(cls_dim)
        self.cat= cat
        self.exchange= exchange
        self.cls_prior= Variable(torch.FloatTensor(cfg.TRAIN.BATCH_SIZE, cfg.GAN.MANIFD_DIM))  ## FOR TEST
        self.cls_prior= self.cls_prior.cuda()
        if cfg.TREE.BRANCH_NUM > 0:
            if use_cap:
                if exchange:
                    self.h_net1 = COND_INIT_STAGE_G_Exchange_Cap(self.gf_dim* 16)
                else:
                    self.h_net1 = COND_INIT_STAGE_G_withCap(self.gf_dim * 16)
            else:
                self.h_net1 = COND_INIT_STAGE_G(self.gf_dim * 16)
            self.img_net1 = share_Gs[0]
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim)
            self.img_net2 = share_Gs[1]
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(self.gf_dim // cfg.TREE.SCALE)
            self.img_net3 = share_Gs[2]

    def forward(self, noise, sen, cls=None, cls_prior=None):
        '''cond: text embedding or class onehot'''        
        c_code1, mu1, logvar1, std1= self.ca_net1(sen)        
        if self.training or (not cfg.TEST.CLS_PRIOR):
            # c_code2, mu2, logvar2, std2= self.ca_net2(cls)
            c_code2, mu2, logvar2, std2= self.vc_net2(noise, cls)  ##{7/25} noise - uncontrollable still
        else:
            if cls_prior is not None:
                c_code2, mu2, logvar2, std2= cls_prior, 0, 0, 0
            else:
                # self.cls_prior.cuda() ##!!! must assign it
                # print('*****************set cuda')
                c_code2, mu2, logvar2, std2= self.cls_prior.data.normal_(0, 1), 0, 0, 0
        
        ## ************** cat z1 with z2 ******************
        if not self.exchange:
            if self.cat== 'concat':
                c_code= torch.cat((c_code1, c_code2), 1)
            elif self.cat== 'product':
                # assert False, 'todo'
                c_code= c_code1* c_code2
            elif self.cat== 'sum':
                # assert False, 'todo'
                c_code= c_code1+ c_code2            
        else:
            if self.cat== 'concat':
                c_code= torch.cat((c_code1, c_code2), 1)
        h_codes = []

        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(c_code, noise)  ## noise concat with z
            h_codes.append(h_code1)
            # fake_img1 = self.img_net1(h_code1)
            # fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            h_codes.append(h_code2)
            # fake_img2 = self.img_net2(h_code2)
            # fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3 = self.h_net3(h_code2, c_code)
            h_codes.append(h_code3)
            # fake_img3 = self.img_net3(h_code3)
            # fake_imgs.append(fake_img3)

        return h_codes, mu1, mu2, logvar1, logvar2, std1, std2

    def get_image(self, entity_hcodes, sen_hcodes):
        fake_imgs= []

        if cfg.TREE.BRANCH_NUM > 0:
            img_code1 = entity_hcodes[0] * sen_hcodes[0]  ## e-wise multiply
            fake_img1 = self.img_net1(img_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            img_code2 = entity_hcodes[1] * sen_hcodes[1]
            fake_img2 = self.img_net2(img_code2)
            fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            img_code3 = entity_hcodes[2] * sen_hcodes[2]            
            fake_img3 = self.img_net3(img_code3)
            fake_imgs.append(fake_img3)

        return fake_imgs

    def image(self, hcodes):
        fake_imgs= []

        if cfg.TREE.BRANCH_NUM > 0:
            img_code1 = hcodes[0]
            fake_img1 = self.img_net1(img_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            img_code2 = hcodes[1]
            fake_img2 = self.img_net2(img_code2)
            fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            img_code3 = hcodes[2]
            fake_img3 = self.img_net3(img_code3)
            fake_imgs.append(fake_img3)

        return fake_imgs


### Build generator by split two z
class COND_G_NET_CATZ(nn.Module):  ## entity_gnet
    def __init__(self, sen_dim, cls_dim, share_Gs, use_cap= False, cat= 'concat', exchange= False):
        super(COND_G_NET_CATZ, self).__init__()
        self.gf_dim= cfg.GAN.GF_DIM
        self.vc_net1= VC_NET(sen_dim)
        # self.ca_net1= CA_NET()  ## INSTEAD OF VC IN SENTENCE inference
        self.vc_net2= VC_NET(cls_dim)
        self.cat= cat
        self.exchange= exchange
        if cfg.TREE.BRANCH_NUM > 0:
            if use_cap:
                if exchange:
                    self.h_net1 = COND_INIT_STAGE_G_Exchange_Cap(self.gf_dim* 16)
                else:
                    self.h_net1 = COND_INIT_STAGE_G_withCap(self.gf_dim * 16)
            else:
                self.h_net1 = COND_INIT_STAGE_G(self.gf_dim * 16)
            self.img_net1 = share_Gs[0]
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim)
            self.img_net2 = share_Gs[1]
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(self.gf_dim // cfg.TREE.SCALE)
            self.img_net3 = share_Gs[2]

    def forward(self, noise, sen, cls):
        '''cond: text embedding or class onehot'''
        c_code1, mu1, logvar1, std1= self.vc_net1(noise, sen)
        # c_code1, mu1, logvar1, std1= self.ca_net1(sen)
        c_code2, mu2, logvar2, std2= self.vc_net2(noise, cls)
        ## ************** cat z1 with z2 ******************
        if not self.exchange:
            if self.cat== 'concat':
                c_code= torch.cat((c_code1, c_code2), 1)
                # c_code= torch.cat((c_code1, c_code2, noise), 1)  ## USE CA
            elif self.cat== 'product':
                # assert False, 'todo'
                c_code= c_code1* c_code2
            elif self.cat== 'sum':
                # assert False, 'todo'
                c_code= c_code1+ c_code2            
        else:
            if self.cat== 'concat':
                c_code= torch.cat((c_code1, c_code2), 1)
        h_codes = []

        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(c_code)
            h_codes.append(h_code1)
            # fake_img1 = self.img_net1(h_code1)
            # fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            h_codes.append(h_code2)
            # fake_img2 = self.img_net2(h_code2)
            # fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3 = self.h_net3(h_code2, c_code)
            h_codes.append(h_code3)
            # fake_img3 = self.img_net3(h_code3)
            # fake_imgs.append(fake_img3)

        return h_codes, mu1, mu2, logvar1, logvar2, std1, std2

    def get_image(self, entity_hcodes, sen_hcodes):
        fake_imgs= []

        if cfg.TREE.BRANCH_NUM > 0:
            img_code1 = entity_hcodes[0] * sen_hcodes[0]  ## e-wise multiply
            fake_img1 = self.img_net1(img_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            img_code2 = entity_hcodes[1] * sen_hcodes[1]
            fake_img2 = self.img_net2(img_code2)
            fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            img_code3 = entity_hcodes[2] * sen_hcodes[2]            
            fake_img3 = self.img_net3(img_code3)
            fake_imgs.append(fake_img3)

        return fake_imgs

    def image(self, hcodes):
        fake_imgs= []

        if cfg.TREE.BRANCH_NUM > 0:
            img_code1 = hcodes[0]
            fake_img1 = self.img_net1(img_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            img_code2 = hcodes[1]
            fake_img2 = self.img_net2(img_code2)
            fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            img_code3 = hcodes[2]
            fake_img3 = self.img_net3(img_code3)
            fake_imgs.append(fake_img3)

        return fake_imgs


### Build generator
class COND_G_NET(nn.Module):  ## entity_gnet
    def __init__(self, cond_dim, share_Gs, use_cap= False):
        super(COND_G_NET, self).__init__()
        self.gf_dim= cfg.GAN.GF_DIM
        self.vc_net= VC_NET(cond_dim)
        if cfg.TREE.BRANCH_NUM > 0:
            if use_cap:
                self.h_net1 = COND_INIT_STAGE_G_withCap(self.gf_dim * 16)
            else:
                self.h_net1 = COND_INIT_STAGE_G(self.gf_dim * 16)
            self.img_net1 = share_Gs[0]
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim)
            self.img_net2 = share_Gs[1]
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(self.gf_dim // cfg.TREE.SCALE)
            self.img_net3 = share_Gs[2]

    def forward(self, noise, cond):
        '''cond: text embedding or class onehot'''
        c_code, mu, logvar, std= self.vc_net(noise, cond)
        h_codes = []

        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(c_code)
            h_codes.append(h_code1)
            # fake_img1 = self.img_net1(h_code1)
            # fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            h_codes.append(h_code2)
            # fake_img2 = self.img_net2(h_code2)
            # fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3 = self.h_net3(h_code2, c_code)
            h_codes.append(h_code3)
            # fake_img3 = self.img_net3(h_code3)
            # fake_imgs.append(fake_img3)

        return h_codes, mu, logvar, std

    def get_image(self, entity_hcodes, sen_hcodes):
        fake_imgs= []

        if cfg.TREE.BRANCH_NUM > 0:
            img_code1 = entity_hcodes[0] * sen_hcodes[0]  ## e-wise multiply
            fake_img1 = self.img_net1(img_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            img_code2 = entity_hcodes[1] * sen_hcodes[1]
            fake_img2 = self.img_net2(img_code2)
            fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            img_code3 = entity_hcodes[2] * sen_hcodes[2]            
            fake_img3 = self.img_net3(img_code3)
            fake_imgs.append(fake_img3)

        return fake_imgs

    def image(self, hcodes):
        fake_imgs= []

        if cfg.TREE.BRANCH_NUM > 0:
            img_code1 = hcodes[0]
            fake_img1 = self.img_net1(img_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            img_code2 = hcodes[1]
            fake_img2 = self.img_net2(img_code2)
            fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            img_code3 = hcodes[2]
            fake_img3 = self.img_net3(img_code3)
            fake_imgs.append(fake_img3)

        return fake_imgs


class G_NET(nn.Module):
    def __init__(self, share_Gs):
        super(G_NET, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.define_module(share_Gs)

    def define_module(self, share_Gs):
        if cfg.GAN.B_CONDITION:
            self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(self.gf_dim * 16)
            self.img_net1 = share_Gs[0]
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim)
            self.img_net2 = share_Gs[1]
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(self.gf_dim // cfg.TREE.SCALE)
            self.img_net3 = share_Gs[2]

    def forward(self, z_code, text_embedding=None):
        if cfg.GAN.B_CONDITION and text_embedding is not None:
            c_code, mu, logvar = self.ca_net(text_embedding)
        else:
            c_code, mu, logvar = z_code, None, None
        h_codes = []

        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)  # if no cond, c_code(z_code) not used
            h_codes.append(h_code1)
            # fake_img1 = self.img_net1(h_code1)
            # fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            h_codes.append(h_code2)
            # fake_img2 = self.img_net2(h_code2)
            # fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3 = self.h_net3(h_code2, c_code)
            h_codes.append(h_code3)
            # fake_img3 = self.img_net3(h_code3)
            # fake_imgs.append(fake_img3)        

        return h_codes, mu, logvar

    def get_image(self, entity_hcodes, sen_hcodes):
        fake_imgs= []

        if cfg.TREE.BRANCH_NUM > 0:
            img_code1 = entity_hcodes[0] * sen_hcodes[0]
            fake_img1 = self.img_net1(img_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            img_code2 = entity_hcodes[1] * sen_hcodes[1]
            fake_img2 = self.img_net2(img_code2)
            fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            img_code3 = entity_hcodes[2] * sen_hcodes[2]            
            fake_img3 = self.img_net3(img_code3)
            fake_imgs.append(fake_img3)

        return fake_imgs


# ############## D networks ################################################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


class ENTITY_D_NET64(nn.Module):
    def __init__(self):
        super(ENTITY_D_NET64, self).__init__()
        self.df_dim= cfg.GAN.DF_DIM
        self.entity_num= cfg.GAN.ENTITY_DIM

        self.downsp_16= encode_image_by_16times(self.df_dim)
        self.fc_real= nn.Linear(self.df_dim*8 *4*4, 1)
        self.fc_ac= nn.Linear(self.df_dim*8 *4*4, self.entity_num+1)

    def forward(self, x):
        x_code= self.downsp_16(x)
        # -> 8ndf x4x4
        x_code= x_code.view(-1, self.df_dim*8 *4*4)
        tp= self.fc_real(x_code)[:,0]
        cp= F.log_softmax(self.fc_ac(x_code))

        return tp, cp


# For 64 x 64 images
class D_NET64(nn.Module):
    def __init__(self):
        super(D_NET64, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),  # fully conv, output1 for matching
            nn.Sigmoid())  ## inverse logit function = logistic function = sigmoid

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),  # another fully conv, output2 for reality
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


### Joint entity D and sentence D
class JOINT_D_NET64(nn.Module):
    def __init__(self, use_cap= False):
        super(JOINT_D_NET64, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM * 2 ####### *** for concat-z ***
        if cfg.TRAIN.CAT_Z!= 'concat':
            self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.entity_num= cfg.GAN.ENTITY_DIM
        self.use_cap= use_cap
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)

        ## first path for sentence
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),  # fully conv, output1 for matching
            nn.Sigmoid())  ## inverse logit function = logistic function = sigmoid
                
        ## second path for class        
        if self.use_cap:
            self.fc_ac_cap= nn.Sequential(
                capsule.CapsuleLinear(out_capsules= self.entity_num+1, in_length=ndf*8, out_length= 16, in_capsules= None),
            )
        else:
            self.fc_ac= nn.Linear(ndf*8 *4*4, self.entity_num+1)

        ## third for reality
        self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),  # another fully conv, output2 for reality
                nn.Sigmoid())

    def forward(self, x_var, c_code):
        x_code = self.img_code_s16(x_var)

        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)
        sen_match = self.logits(h_c_code)

        real = self.uncond_logits(x_code)
                        
        if self.use_cap:
            x_code= x_code.permute(0, 2, 3, 1)  ## depth dim as feature/capsule
            x_code= x_code.contiguous().view(-1, 4*4, self.df_dim*8)
            out= self.fc_ac_cap(x_code)
            cp= out.norm(dim=-1)
            cp= F.log_softmax(cp)
            # cp= cp.log()
        else:
            x_code= x_code.view(-1, self.df_dim*8 *4*4)        
            cp= F.log_softmax(self.fc_ac(x_code))
            
        return [sen_match.view(-1), real.view(-1), cp]        


class ENTITY_D_NET128(nn.Module):
    def __init__(self):
        super(ENTITY_D_NET128, self).__init__()
        self.df_dim= cfg.GAN.DF_DIM
        self.entity_num= cfg.GAN.ENTITY_DIM

        self.downsp_16= encode_image_by_16times(self.df_dim)
        self.downsp_32= downBlock(self.df_dim*8, self.df_dim*16)
        self.downsp_32_1= Block3x3_leakRelu(self.df_dim*16, self.df_dim*8)  ## 8ndf x4x4
        
        self.fc_real= nn.Linear(self.df_dim*8 *4*4, 1)
        self.fc_ac= nn.Linear(self.df_dim*8 *4*4, self.entity_num+1)

    def forward(self, x):
        x_code= self.downsp_16(x)
        x_code= self.downsp_32(x_code)
        x_code= self.downsp_32_1(x_code) ## -> 8ndf x4x4
        
        x_code= x_code.view(-1, self.df_dim*8 *4*4)
        tp= self.fc_real(x_code)[:,0]
        cp= F.log_softmax(self.fc_ac(x_code))

        return tp, cp


# For 128 x 128 images
class D_NET128(nn.Module):
    def __init__(self):
        super(D_NET128, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)  # ->4x4
        x_code = self.img_code_s32_1(x_code)  # ->ndf*8 x4x4. huanyuan

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)  ## disc hidden state
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


### Joint entity D and sentence D
class JOINT_D_NET128(nn.Module):
    def __init__(self, use_cap= False):
        super(JOINT_D_NET128, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM * 2 ####### *** for concat-z ***
        if cfg.TRAIN.CAT_Z!= 'concat':
            self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.entity_num= cfg.GAN.ENTITY_DIM
        self.use_cap= use_cap
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)        
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)        

        ## first path for sentence
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),  # fully conv, output1 for matching
            nn.Sigmoid())  ## inverse logit function = logistic function = sigmoid
                
        ## second path for class
        if self.use_cap:
            self.fc_ac_cap= nn.Sequential(
                # Reshape (-1, ndf*8, 16),  ## make it 3d
                capsule.CapsuleLinear(out_capsules= self.entity_num+1, in_length=ndf*8, out_length= 16, in_capsules= None),
                # Reshape(-1, ngf*4*4* 2),
                # nn.BatchNorm1d(ngf*4*4* 2),
                # GLU()
            )
        else:
            self.fc_ac= nn.Linear(ndf*8 *4*4, self.entity_num+1)

        ## third for reality
        self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),  # another fully conv, output2 for reality
                nn.Sigmoid())

    def forward(self, x_var, c_code):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)        
        x_code = self.img_code_s32_1(x_code)        

        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)
        sen_match = self.logits(h_c_code)

        real = self.uncond_logits(x_code)
        
        if self.use_cap:
            x_code= x_code.permute(0, 2, 3, 1)  ## depth dim as feature/capsule
            x_code= x_code.contiguous().view(-1, 4*4, self.df_dim*8)
            out= self.fc_ac_cap(x_code)
            cp= out.norm(dim=-1)
            cp= F.log_softmax(cp)
            # cp= cp.log()
        else:
            x_code= x_code.view(-1, self.df_dim*8 *4*4)        
            cp= F.log_softmax(self.fc_ac(x_code))
            
        return [sen_match.view(-1), real.view(-1), cp]


class ENTITY_D_NET256(nn.Module):
    def __init__(self):
        super(ENTITY_D_NET256, self).__init__()
        self.df_dim= cfg.GAN.DF_DIM
        self.entity_num= cfg.GAN.ENTITY_DIM

        self.downsp_16= encode_image_by_16times(self.df_dim)
        self.downsp_32= downBlock(self.df_dim*8, self.df_dim*16)
        self.downsp_64= downBlock(self.df_dim*16, self.df_dim*32)  ## 4x4
        self.downsp_64_1= Block3x3_leakRelu(self.df_dim*32, self.df_dim*16)  ## 16ndf x4x4
        self.downsp_64_2= Block3x3_leakRelu(self.df_dim*16, self.df_dim*8)  ## 8ndf x4x4
        
        self.fc_real= nn.Linear(self.df_dim*8 *4*4, 1)
        self.fc_ac= nn.Linear(self.df_dim*8 *4*4, self.entity_num+1)

    def forward(self, x):
        x_code= self.downsp_16(x)
        x_code= self.downsp_32(x_code)
        x_code= self.downsp_64(x_code)
        x_code= self.downsp_64_1(x_code)
        x_code= self.downsp_64_2(x_code)
        
        x_code= x_code.view(-1, self.df_dim*8 *4*4)
        tp= self.fc_real(x_code)[:,0]
        cp= F.log_softmax(self.fc_ac(x_code))

        return tp, cp


# For 256 x 256 images
class D_NET256(nn.Module):
    def __init__(self):
        super(D_NET256, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)  # 8ndf x4x4, huanyuan

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


### Joint entity D and sentence D
class JOINT_D_NET256(nn.Module):
    def __init__(self):
        super(JOINT_D_NET256, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.entity_num= cfg.GAN.ENTITY_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)  # 8ndf x4x4, huanyuan

        ## first path for sentence
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),  # fully conv, output1 for matching
            nn.Sigmoid())  ## inverse logit function = logistic function = sigmoid
                
        ## second path for class
        self.fc_ac= nn.Linear(ndf*8 *4*4, self.entity_num+1)

        ## third for reality
        self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),  # another fully conv, output2 for reality
                nn.Sigmoid())

    def forward(self, x_var, c_code):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)

        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)
        sen_match = self.logits(h_c_code)

        real = self.uncond_logits(x_code)
                
        x_code= x_code.view(-1, self.df_dim*8 *4*4)        
        cp= F.log_softmax(self.fc_ac(x_code))
            
        return [sen_match.view(-1), real.view(-1), cp]


# For 512 x 512 images: Recommended structure, not test yet
class D_NET512(nn.Module):
    def __init__(self):
        super(D_NET512, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s128 = downBlock(ndf * 32, ndf * 64)
        self.img_code_s128_1 = Block3x3_leakRelu(ndf * 64, ndf * 32)
        self.img_code_s128_2 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s128_3 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s128(x_code)
        x_code = self.img_code_s128_1(x_code)
        x_code = self.img_code_s128_2(x_code)
        x_code = self.img_code_s128_3(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


# For 1024 x 1024 images: Recommended structure, not test yet
class D_NET1024(nn.Module):
    def __init__(self):
        super(D_NET1024, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s128 = downBlock(ndf * 32, ndf * 64)
        self.img_code_s256 = downBlock(ndf * 64, ndf * 128)
        self.img_code_s256_1 = Block3x3_leakRelu(ndf * 128, ndf * 64)
        self.img_code_s256_2 = Block3x3_leakRelu(ndf * 64, ndf * 32)
        self.img_code_s256_3 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s256_4 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s128(x_code)
        x_code = self.img_code_s256(x_code)
        x_code = self.img_code_s256_1(x_code)
        x_code = self.img_code_s256_2(x_code)
        x_code = self.img_code_s256_3(x_code)
        x_code = self.img_code_s256_4(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]
