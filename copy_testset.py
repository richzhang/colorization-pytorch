
import os
import sys
from util import util
import numpy as np

# original imagenet
orig_path = '/data/big/dataset/ILSVRC2012/val/'

# new dataset
new_path = './dataset/ilsvrc2012/val/imgs'
util.mkdirs(new_path)

val_inds = np.load('./ilsvrclin12_val_inds.npy')

for val_ind in val_inds:
	os.symlink('%s/ILSVRC2012_val_%08d.JPEG'%(orig_path,val_ind+1),
		'%s/ILSVRC2012_val_%08d.JPEG'%(new_path,val_ind+1))
