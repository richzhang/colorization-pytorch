
import os
import sys
from util import util
import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_path', type=str, default='/data/big/dataset/ILSVRC2012')
parser.add_argument('--out_path', type=str, default='./dataset/ilsvrc2012/')

opt = parser.parse_args()
orig_path = opt.in_path
print('Copying ILSVRC from...[%s]'%orig_path)

# Copy over whole training set
trn_path = os.path.join(opt.out_path,'train')
util.mkdirs(opt.out_path)
os.symlink(os.path.join(opt.in_path,'train'),trn_path)
print('Making training set in...[%s]'%trn_path)

# Copy over subset of ILSVRC12 val set for colorization val set
val_path = os.path.join(opt.out_path,'val/imgs')
util.mkdirs(val_path)
print('Making validation set in...[%s]'%val_path)
for val_ind in range(1000):
	os.symlink('%s/ILSVRC2012_val_%08d.JPEG'%(orig_path,val_ind+1),
		'%s/ILSVRC2012_val_%08d.JPEG'%(val_path,val_ind+1))

# Copy over subset of ILSVRC12 val set for colorization test set
test_path = os.path.join(opt.out_path,'test/imgs')
util.mkdirs(test_path)
val_inds = np.load('./resources/ilsvrclin12_val_inds.npy')
print('Making test set in...[%s]'%test_path)
for val_ind in val_inds:
	os.symlink('%s/ILSVRC2012_val_%08d.JPEG'%(orig_path,val_ind+1),
		'%s/ILSVRC2012_val_%08d.JPEG'%(test_path,val_ind+1))
