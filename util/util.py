from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from IPython import embed

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) ),0, 1) * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)





# Color conversion code
def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    return torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]

    rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    return rgb

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])

    return torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

def lab2xyz(lab):
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(out.is_cuda):
        sc = sc.cuda()

    out = out*sc
    return out

def rgb2lab(rgb, l_norm=100, ab_norm=110, l_cent=50):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:,[0],:,:]-l_cent)/l_norm
    ab_rs = lab[:,1:,:,:]/ab_norm
    return torch.cat((l_rs,ab_rs),dim=1)

def lab2rgb(lab_rs, l_norm=100, ab_norm=110, l_cent=50):
    l = lab_rs[:,[0],:,:]*l_norm + l_cent
    ab = lab_rs[:,1:,:,:]*ab_norm
    lab = torch.cat((l,ab),dim=1)

    return xyz2rgb(lab2xyz(lab))

def get_colorization_data(data_raw, l_norm=100, ab_norm=110, l_cent=50, mask_cent=.5, ab_thresh=5.,
    p=.125):
    data = {}
    # data['A_paths'] = ''
    # data['B_paths'] = ''

    # print(H,W,Hnew,Wnew,h,w)
    data_lab = rgb2lab(data_raw[0], l_norm=100, ab_norm=110, l_cent=50, )
    data['A'] = data_lab[:,[0,],:,:]
    data['B'] = data_lab[:,1:,:,:]

    if(ab_thresh > 0): # mask out grayscale images
        thresh = 1.*ab_thresh/ab_norm
        mask = torch.sum(torch.abs(torch.max(torch.max(data['B'],dim=3)[0],dim=2)[0]-torch.min(torch.min(data['B'],dim=3)[0],dim=2)[0]),dim=1) >= thresh
        data['A'] = data['A'][mask,:,:,:]
        data['B'] = data['B'][mask,:,:,:]
        # print('Removed %i points'%torch.sum(mask==0).numpy())
        if(torch.sum(mask)==0):
            return None

    return add_color_points(data, p=p, mask_cent=mask_cent)

def add_color_points(data,p=.125,Ps=[1,2,3,4,5,6,7,8,9,], mask_cent=.5):
# def add_color_points(data,p=.125,Ps=[30,]):
    # print(data['B'].shape)
    N,C,H,W = data['B'].shape

    data['hint_B'] = torch.zeros_like(data['B'])
    data['mask_B'] = torch.zeros_like(data['A'])

    for nn in range(N):
        while(np.random.rand() < (1-p) ):
            P = np.random.choice(Ps) # patch size
            # h = np.random.randint(H-P+1)
            # w = np.random.randint(W-P+1)
            h = int(np.clip(np.random.normal( (H-P+1)/2., (H-P+1)/4.), 0, H-P))
            w = int(np.clip(np.random.normal( (W-P+1)/2., (W-P+1)/4.), 0, W-P))

            data['hint_B'][nn,:,h:h+P,w:w+P] = torch.mean(torch.mean(data['B'][nn,:,h:h+P,w:w+P],dim=2,keepdim=True),dim=1,keepdim=True).view(1,C,1,1)
            # data['hint_B'][nn,:,h:h+P,w:w+P] = data['B'][nn,:,h:h+P,w:w+P]
            data['mask_B'][nn,:,h:h+P,w:w+P] = 1

    data['mask_B']-=mask_cent

    return data

def add_color(data,mask,mask_cent=.5,h=128,w=128,P=1,a=0,b=0):
    data[:,0,h:h+P,w:w+P] = a
    data[:,1,h:h+P,w:w+P] = b
    mask[:,:,h:h+P,w:w+P] = 1-mask_cent

    return (data,mask)

def crop_mult(data,mult=16,Hmax=800,Wmax=1200):
    # crop to multiple of 4
    H,W = data.shape[2:]
    # embed()
    Hnew = int(min(H/mult*mult,Hmax))
    Wnew = int(min(W/mult*mult,Wmax))
    h = (H-Hnew)/2
    w = (W-Wnew)/2

    return data[:,:,h:h+Hnew,w:w+Wnew]

def encode_ab_ind(data_ab, ab_norm=110., ab_max=110., ab_quant=10.):
    # encode ab value into an index
    # data_ab   Nx2xHxW \in [-1, 1]
    # data_q    Nx1xHxW \in [0,Q)

    # embed()
    A = 2*ab_max/ab_quant + 1 # number of bins
    data_ab_rs = torch.round((data_ab*ab_norm + ab_max)/ab_quant) # normalized bin number
    data_q = data_ab_rs[:,[0],:,:]*A + data_ab_rs[:,[1],:,:]
    return data_q

def decode_ind_ab(data_q, ab_norm=110., ab_max=110., ab_quant=10.):
    # decode index into ab value
    A = 2*ab_max/ab_quant + 1

    data_a = data_q/A
    data_b = data_q - data_a*A
    data_ab = torch.cat((data_a[:,None,:,:],data_b[:,None,:,:]),dim=1)

    data_ab = ((data_ab.type(torch.cuda.FloatTensor)*ab_quant) - ab_max)/ab_norm

    return data_ab

def decode_max_ab(data_ab_quant, ab_norm=110., ab_max=110., ab_quant=10.):
    # data_quant NxQxHxW \in [0,1]
    # data_ab   Nx2xHxW \in [-1, 1]
    # embed()
    data_q = torch.argmax(data_ab_quant,dim=1)
    return decode_ind_ab(data_q)
    # data_a_rs = torch.floor(data_q/A)
    # data_b_rs = data_q - data_a_rs*A
    # data_ab_rs = torch.cat((data_a_rs,data_b_rs),dim=1)
    # data_ab = ((data_ab_rs*ab_quant) - ab_max)/ab_norm

    # return data_ab

def decode_mean(data_ab_quant, ab_norm=110., ab_max=110., ab_quant=10.):
    # data_quant NxQxHxW \in [0,1]
    # data_ab_inf Nx2xHxW \in [-110, 110]

    (N,Q,H,W) = data_ab_quant.shape
    A = 2*ab_max/ab_quant + 1
    a_range = torch.range(-ab_max, ab_max, step=ab_quant)[None,:,None,None]
    a_range = a_range.cuda()

    # reshape to AB space
    data_ab_quant = data_ab_quant.view((N,A,A,H,W))
    data_a_total = torch.sum(data_ab_quant,dim=2)
    data_b_total = torch.sum(data_ab_quant,dim=1)

    # matrix multiply
    data_a_inf = torch.sum(data_a_total * a_range,dim=1,keepdim=True)
    data_b_inf = torch.sum(data_b_total * a_range,dim=1,keepdim=True)

    data_ab_inf = torch.cat((data_a_inf,data_b_inf),dim=1)/ab_norm
    # embed()

    return data_ab_inf
