import os
from skimage import io, color
import numpy as np

folder_path = ["annotations/training", "annotations/validation"]
to_label = True

def _to_label(image):
    # 自建数据集常见问题
    # 需要将4通道png图像转换为1通道png图像
    # https://github.com/open-mmlab/mmsegmentation/issues/2471
    # 代码来自:
    # https://github.com/open-mmlab/mmsegmentation/blob/
    # 9b8e8b730c2a17e9f0517489cb707e981e0eab39/tools/
    # dataset_converters/vaihingen.py#L68-L80
    h, w, c = image.shape
    color_map = np.array([[0, 0, 0],
                        [128, 0, 0],
                        [0, 128, 0]])
    flatten_v = np.matmul(
        image.reshape(-1, c),
        np.array([2, 3, 4]).reshape(3, 1))
    out = np.zeros_like(flatten_v)
    for idx, class_color in enumerate(color_map):
        value_idx = np.matmul(class_color,
                            np.array([2, 3, 4]).reshape(3, 1))
        out[flatten_v == value_idx] = idx
    return out.reshape(h, w)

for f_path in folder_path:
    for filename in os.listdir(f_path):  # 遍历文件夹下的所有png文件
        if filename.endswith(".png"):
            image_path = os.path.join(f_path, filename)

            # 使用skimage库读取RGB图像
            image = io.imread(image_path)
            if image.shape[-1] == 4:
                image = image[:, :, :3]

            if to_label:
                gray_image = _to_label(image)
                gray_image = gray_image.astype('uint8')
            else:
                gray_image = color.rgb2gray(image[:, :, :3])
                # 将灰度图像转为uint8格式
                gray_image = (gray_image * 255).astype('uint8')

            # 保存灰度图像（替换原文件）
            io.imsave(image_path, gray_image)

"""
to_label=True时,
目的是得到像素值以1为步长从0递增的整数作为真值标签.
那么必须将灰度图像先转换为uint8格式再保存,
否则在保存图像时像素值会被自动缩放到0-255之间.
"""