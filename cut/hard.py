# -*- coding: utf-8 -*-
import os
import ssl
import numpy as np
import cv2
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

# ---------------------------------------------------------------
### Load Test Dataset
class TestDataset(BaseDataset):
    def __init__(self, images_dirs, preprocessing=None):
        self.images_fps = []
        for images_dir in images_dirs:
            self.images_fps += [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        if image is None:
            raise FileNotFoundError(f"Image not found: {self.images_fps[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image, self.images_fps[i]  # 返回图像及其路径

    def __len__(self):
        return len(self.images_fps)

# 对数据集预处理的函数
def to_tensor(x, **kwargs):
    if x.ndim == 2:
        return np.expand_dims(x, axis=0).astype('float32')
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Resize(height=512, width=512),
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)

# ---------------------------------------------------------------
### Save Prediction Function

def save_prediction_only(mask_pred, save_dirs, image_path):
    for save_dir in save_dirs:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # 根据图像来源确定保存路径
    original_filename = os.path.basename(image_path).split('.')[0]  # 取出原始文件名
    if 'dataset/0' in image_path:  # 第一个目录
        save_path = os.path.join(save_dirs[0], f"{original_filename}.png")
    elif 'dataset/htr1' in image_path:  # 第二个目录
        save_path = os.path.join(save_dirs[1], f"{original_filename}.png")
    #elif '/CRVO_combined' in image_path:  # 第三个目录
        #save_path = os.path.join(save_dirs[2], f"{original_filename}.png")
    else:
        raise ValueError(f"Unknown : {image_path}")

    # 将掩码值缩放到 [0, 255] 并保存
    mask_pred = (mask_pred.squeeze() * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_pred)

# ---------------------------------------------------------------
### Test Model
'''test_dirs不用换 save_dirs不用换'''
if __name__ == '__main__':
    test_dirs = [
        '/media/disk/Backup/03drive/02LMW/from02drive/htr_final/dataset/0',  # 0类数据
        '/media/disk/Backup/03drive/02LMW/from02drive/htr_final/dataset/htr1',  # 1类数据
    ]
    save_dirs = [
        '/media/disk/Backup/03drive/02LMW/from02drive/htr_end/attention/hard/0',  # 0类保存目录
        '/media/disk/Backup/03drive/02LMW/from02drive/htr_end/attention/hard/1',  # 1类保存目录
    ]
    for save_dir in save_dirs:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    ENCODER = 'vgg16'
    ssl._create_default_https_context = ssl._create_unverified_context

    DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained model
    #model = torch.load('/media/disk/Backup/03drive/03SJY/second_segmentation/albumeneations/model/hard_exduates.pth')
    #model = torch.load('/media/disk/Backup/03drive/03SJY/re_hd/vgg16_2/new/model.pth')
    model = torch.load('/media/disk/Backup/03drive/03SJY/second_segmentation/no_attention/model/hard_exduates.pth')
    model.to(DEVICE)
    model.eval()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')

    test_dataset = TestDataset(
        test_dirs,
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2)

    with torch.no_grad():
        for i, (image, image_path) in enumerate(test_loader):
            image = image.to(DEVICE)
            mask_pred = model(image)

            mask_pred = mask_pred.detach().cpu().numpy()

            # Save predictions for each image
            for j in range(mask_pred.shape[0]):
                save_prediction_only(mask_pred[j], save_dirs, image_path[j])

    print("complete.")
