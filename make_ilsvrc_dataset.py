
import os
from util import util
import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_path', type=str, default='/data/big/dataset/ILSVRC2012')
parser.add_argument('--out_path', type=str, default='./dataset/ilsvrc2012/')

opt = parser.parse_args()
orig_path = opt.in_path
print('Copying ILSVRC from...[%s]' % orig_path)

# Copy over part of training set (for initializer)
trn_small_path = os.path.join(opt.out_path, 'train_small')
util.mkdirs(opt.out_path)
util.mkdirs(trn_small_path)
train_subdirs = os.listdir(os.path.join(opt.in_path, 'train'))
for train_subdir in train_subdirs[:10]:
    os.symlink(os.path.join(opt.in_path, 'train', train_subdir), os.path.join(trn_small_path, train_subdir))
print('Making small training set in...[%s]' % trn_small_path)

# Copy over whole training set
trn_path = os.path.join(opt.out_path, 'train')
util.mkdirs(opt.out_path)
os.symlink(os.path.join(opt.in_path, 'train'), trn_path)
print('Making training set in...[%s]' % trn_path)

# Copy over subset of ILSVRC12 val set for colorization val set
val_path = os.path.join(opt.out_path, 'val/imgs')
util.mkdirs(val_path)
print('Making validation set in...[%s]' % val_path)
for val_ind in range(1000):
    os.system('ln -s %s/val/ILSVRC2012_val_%08d.JPEG %s/ILSVRC2012_val_%08d.JPEG' % (orig_path, val_ind + 1, val_path, val_ind + 1))
    # os.system('cp %s/val/ILSVRC2012_val_%08d.JPEG %s/ILSVRC2012_val_%08d.JPEG'%(orig_path,val_ind+1,val_path,val_ind+1))

# Copy over subset of ILSVRC12 val set for colorization test set
test_path = os.path.join(opt.out_path, 'test/imgs')
util.mkdirs(test_path)
val_inds = np.load('./resources/ilsvrclin12_val_inds.npy')
print('Making test set in...[%s]' % test_path)
for val_ind in val_inds:
    os.system('ln -s %s/val/ILSVRC2012_val_%08d.JPEG %s/ILSVRC2012_val_%08d.JPEG' % (orig_path, val_ind + 1, test_path, val_ind + 1))
    # os.system('cp %s/val/ILSVRC2012_val_%08d.JPEG %s/ILSVRC2012_val_%08d.JPEG'%(orig_path,val_ind+1,test_path,val_ind+1))
