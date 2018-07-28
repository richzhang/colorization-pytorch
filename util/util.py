from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


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
        xyz_scale = xyz_scale.cuda()

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

def rgb2lab(rgb,scale=110.):
    return xyz2lab(rgb2xyz(rgb))/scale

def lab2rgb(lab,scale=110.):
    return xyz2rgb(lab2xyz(scale*lab))

def get_colorization_data(data_raw):
    data = {}
    data['A_paths'] = ''
    data['B_paths'] = ''

    # crop to multiple of 4
    H,W = data_raw[0].shape[2:]
    Hnew = min(H/4*4,800)
    Wnew = min(W/4*4,1200)
    h = (H-Hnew)/2
    w = (W-Wnew)/2
    # print(H,W,Hnew,Wnew,h,w)
    data_lab = rgb2lab(data_raw[0][:,:,h:h+Hnew,w:w+Wnew])
    data['A'] = data_lab[:,[0,],:,:]
    data['B'] = data_lab[:,1:,:,:]

    return add_color_points(data)

def add_color_points(data,p=.125,Ps=[1,2,3,4,5,6,7,8,9]):
    N,C,H,W = data['B'].shape

    data['hint_B'] = 0*data['B']
    data['mask_B'] = 0*data['A']
    # data['hint_B'][:,:,H/2-P:H/2+P+1,W/2-P:W/2+P+1] = data['B'][:,:,H/2-P:H/2+P+1,W/2-P:W/2+P+1]
    # data['mask_B'][:,:,H/2-P:H/2+P+1,W/2-P:W/2+P+1] = 1
    # data['hint_B'] = 1.*data['B']
    # data['mask_B'] = 0*data['A'] + 1

    while(np.random.rand() < (1-p) ):
        P = np.random.choice(Ps) # patch size
        # h = np.random.randint(H-P+1)
        # w = np.random.randint(W-P+1)
        h = int(np.clip(np.random.normal( (H-P+1)/2., (H-P+1)/4.), 0, H-P))
        w = int(np.clip(np.random.normal( (W-P+1)/2., (W-P+1)/4.), 0, W-P))

        data['hint_B'][:,:,h:h+P+1,w:w+P+1] = torch.mean(torch.mean(data['B'][:,:,h:h+P+1,w:w+P+1],dim=2),dim=2).view(N,C,1,1)
        data['mask_B'][:,:,h:h+P+1,w:w+P+1] = 1

    return data


