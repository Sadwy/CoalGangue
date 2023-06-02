import matplotlib.image as mpig
import numpy as np

print('######################')
print('check out train images')
sets = np.array([])
img_list = []
for i in range(1, 200+1):
    img_dir = f'annotations/training/{i}.png'
    img = mpig.imread(img_dir)
    img_list.append(img)
    sets = np.append(sets, np.unique(img))
    # if i in range(0, 200+1, 40):
    #     print(f'{i} images read')

print('the number of train imges is', len(img_list))
print('sets.shape is', sets.shape)
print('np.unique(sets) is', np.unique(sets))
sets0 = sets


print('######################')
print('check out val images')
sets = np.array([])
img_list = []
for i in range(201, 236+1):
    img_dir = f'annotations/validation/{i}.png'
    img = mpig.imread(img_dir)
    img_list.append(img)
    sets = np.append(sets, np.unique(img))
    # if i in range(201, 236+1, 20):
    #     print(f'{i} images read')

print('the number of val imges is', len(img_list))
print('sets.shape is', sets.shape)
print('np.unique(sets) is', np.unique(sets))


print('######################')
if np.unique(sets0) != np.unique(sets):
    raise ValueError('训练集和验证集的标签不一致!')
print('两个np.unique(sets)相等，说明训练集和验证集的标签是一致的.')
