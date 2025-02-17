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
import matplotlib.pyplot as plt
from segmentation_models_pytorch.utils import train

# ---------------------------------------------------------------
### 加载数据集
class Dataset(BaseDataset):
    CLASSES = ['background', 'disease']  # 背景和疾病两个类

    def __init__(self, images_dir, masks_dir, classes=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]#image和mask完整的路径

        valid_image_mask_pairs = []
        for img_fp, mask_fp in zip(self.images_fps, self.masks_fps):
            if os.path.exists(mask_fp):
                valid_image_mask_pairs.append((img_fp, mask_fp))
            
        self.images_fps, self.masks_fps = zip(*valid_image_mask_pairs)  # 保留图像有对应掩码的

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes if cls.lower() in self.CLASSES]
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        #保证能读取到照片，读取不到就会有提示，下面mask同理
        if image is None:
            raise FileNotFoundError(f"image not found {self.images_fps[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self.masks_fps[i]
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"not found {mask_path}")
        mask = np.where(mask > 30, 1, 0).astype('float32') #其实可以不用这个mask>30,因为原来mask就是黑白
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images_fps)


def to_tensor(x, **kwargs):
    if x.ndim == 2:
        return np.expand_dims(x, axis=0).astype('float32')
    return x.transpose(2, 0, 1).astype('float32')
def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Resize(height=256,width=256),
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
# ---------------------------------------------------------------
### 保存结果的函数
def save_prediction(image, mask_true, mask_pred, epoch, batch_idx, img_idx, save_dir="/media/disk/Backup/03drive/03SJY/second_segmentation/no_attention/result/cotton_wool_spots/save"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    image = np.clip(image, 0, 1)

    ax[0].imshow(image.transpose(1, 2, 0))
    ax[0].set_title("Image")
    ax[0].axis('off')

    ax[1].imshow(mask_true.squeeze(), cmap="gray")
    ax[1].set_title("True Mask")
    ax[1].axis('off')

    ax[2].imshow(mask_pred.squeeze(), cmap="gray")
    ax[2].set_title("Predicted")
    ax[2].axis('off')

    save_path = os.path.join(save_dir, f"epoch{epoch}batch{batch_idx}img{img_idx}.png")
    plt.savefig(save_path, dpi=150)  
    plt.close()


class CombinedLoss(torch.nn.Module):
    def __init__(self, dice_loss, focal_loss):
        super(CombinedLoss, self).__init__()
        self.dice_loss = dice_loss
        self.focal_loss = focal_loss
        self.__name__ = "CombinedLoss" 
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return dice + focal


def save_metrics_to_txt(epoch, train_logs, valid_logs, save_path='/media/disk/Backup/03drive/03SJY/second_segmentation/no_attention/result/cotton_wool_spots/table.txt'):
    with open(save_path, 'a') as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Training metrics: IoU: {train_logs['iou_score']}, Accuracy: {train_logs['accuracy']}, Precision: {train_logs['precision']}\n")
        f.write(f"Validation metrics: IoU: {valid_logs['iou_score']}, Accuracy: {valid_logs['accuracy']}, Precision: {valid_logs['precision']}\n")
        f.write("\n")


# ---------------------------------------------------------------
### 创建模型并训练
if __name__ == '__main__':
    x_train_dir = '/media/disk/Backup/03drive/03SJY/cotton2/cotton_wool_spots/Training_Images'
    y_train_dir = '/media/disk/Backup/03drive/03SJY/cotton2/cotton_wool_spots/Training_Labels'
    x_valid_dir = '/media/disk/Backup/03drive/03SJY/cotton2/cotton_wool_spots/Testing_Images'
    y_valid_dir = '/media/disk/Backup/03drive/03SJY/cotton2/cotton_wool_spots/Testing_Labels'
    ENCODER = 'vgg19'
    ssl._create_default_https_context = ssl._create_unverified_context

    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['disease']
    ACTIVATION = 'sigmoid'
    DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
#用smp包创建模型
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=2)
#自定义损失函数
    dice_loss = smp.utils.losses.DiceLoss()  
    focal_loss = smp.losses.FocalLoss(mode='binary')
    combined_loss = CombinedLoss(dice_loss, focal_loss)
#指标
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Accuracy(threshold=0.5),
        smp.utils.metrics.Precision(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=combined_loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=combined_loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_score = 0

    for epoch in range(0, 75):
        print(f"\nEpoch: {epoch}")
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        save_metrics_to_txt(epoch, train_logs, valid_logs)

        # 保存最佳模型
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, '/media/disk/Backup/03drive/03SJY/second_segmentation/no_attention/model/cotton_wool_spots.pth')
            print('Model saved!')

        # 每 10 个 epoch 保存一次预测的结果
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                for i, (image, mask) in enumerate(valid_loader):
                    image = image.to(DEVICE)
                    mask_pred = model(image)
                    
                    mask_true = mask.detach().cpu().numpy()
                    mask_pred = mask_pred.detach().cpu().numpy()

                    # 保存每个图像的预测
                    for j in range(mask_pred.shape[0]):
                        save_prediction(image.detach().cpu().numpy()[j], mask_true[j], mask_pred[j], epoch, i, j)

print('done')
