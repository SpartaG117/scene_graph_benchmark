import os
import numpy as np


dir = '/home/gzx/data/Dataset/image_caption/coco/x152_grid_320x500'
files = os.listdir(dir)

# for f in files:
# #     print(f)
# #     data = np.load(os.path.join(dir, f))['feat']
# #     print(data.shape)
# #     break
# np.savez(os.path.join('/home/gzx/data/Dataset/image_caption/coco/', f), feat=data)
f1 = np.load(os.path.join(dir, '371800.npz'))['feat']
print(f1.shape)
f2 = np.load(os.path.join('/home/gzx/data/Dataset/image_caption/coco/x152_box_feature', '371800.npz'))['feat']
print(f2.shape)