import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# 配置设备
device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

n_classes = 2  # 二分类
epoches = 10  # 训练轮次

# 记录各个指标的列表，用于画图
train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_precisions, val_precisions = [], []
train_recalls, val_recalls = [], []
train_f1s, val_f1s = [], []
train_roc_aucs, val_roc_aucs = [], []


# 数据集定义
class ImageDataset(Dataset):
    def __init__(self, normal_dir, rvo_dir, transform=None, train=True, num_samples_per_class=128):
        # 获取正常和RVO的文件列表
        class_0_images = [os.path.join(normal_dir, img) for img in os.listdir(normal_dir)]
        class_1_images = [os.path.join(rvo_dir, img) for img in os.listdir(rvo_dir)]

        # 随机抽取指定数量的图片
        class_0_images = np.random.choice(class_0_images, size=num_samples_per_class, replace=False)
        class_1_images = np.random.choice(class_1_images, size=num_samples_per_class, replace=False)
        class_0_images = class_0_images.tolist()
        class_1_images = class_1_images.tolist()
        class_0_train, class_0_val = train_test_split(class_0_images, test_size=0.3, random_state=42)
        class_1_train, class_1_val = train_test_split(class_1_images, test_size=0.3, random_state=42)

        # 根据train参数选择使用训练集或验证集
        if train:
            self.all_images = class_0_train + class_1_train
        else:
            self.all_images = class_0_val + class_1_val

        np.random.shuffle(self.all_images)  # 打乱数据顺序

        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 确定标签
        label = 0 if 'copy' in img_path else 1
        return image, label


normal_dir = '/media/disk/Backup/03drive/02LMW/from02drive/htr2/kaggle/normal/normal'
rvo_dir = '/media/disk/Backup/03drive/02LMW/from02drive/htr2/htr'


# 定义图像的变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建训练集和验证集
train_dataset = ImageDataset(normal_dir=normal_dir, rvo_dir=rvo_dir, transform=transform, train=True)
val_dataset = ImageDataset(normal_dir=normal_dir, rvo_dir=rvo_dir, transform=transform, train=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)


# ResNet50 with dropout
class ResNet50WithDropout(nn.Module):
    def __init__(self, n_classes=2, dropout_prob=0.5):
        super(ResNet50WithDropout, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features, n_classes)
        )

    def forward(self, x):
        return self.resnet(x)


# 初始化模型
model = ResNet50WithDropout(n_classes=n_classes, dropout_prob=0.5).to(device)


# 计算指标函数
def calculate_metrics(labels, preds, probs):
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')
    
    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])  # 针对二分类问题
    except ValueError as e:
        print(f"Warning: {e}")
        roc_auc = -1  # 或者设置为其他默认值
    
    return acc, precision, recall, f1, roc_auc


# 保存训练和验证指标
def save_metrics(epoch, train_metrics, val_metrics, save_path):
    with open(save_path, "a") as f:
        f.write(f"Epoch {epoch + 1}\n")
        f.write(
            f"Train Loss: {train_metrics[0]:.5f}, Acc: {train_metrics[1]:.2f}%, Precision: {train_metrics[2]:.4f}, Recall: {train_metrics[3]:.4f}, F1 Score: {train_metrics[4]:.4f}, ROC AUC: {train_metrics[5]:.4f}\n")
        f.write(
            f"Val Loss: {val_metrics[0]:.5f}, Acc: {val_metrics[1]:.2f}%, Precision: {val_metrics[2]:.4f}, Recall: {val_metrics[3]:.4f}, F1 Score: {val_metrics[4]:.4f}, ROC AUC: {val_metrics[5]:.4f}\n")
        f.write("--------------\n")


# 训练函数
def train_model(model, train_loader, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0.
    total_corrects = 0.
    total = 0.
    all_train_labels = []
    all_train_preds = []
    all_train_probs = []

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        probs = nn.Softmax(dim=1)(outputs).detach().cpu().numpy()

        all_train_labels.extend(labels.cpu().numpy())
        all_train_preds.extend(preds.cpu().numpy())
        all_train_probs.extend(probs)

        total_corrects += torch.sum(preds.eq(labels))
        total_loss += loss.item() * inputs.size(0)
        total += labels.size(0)

    total_loss = total_loss / total
    acc = 100 * total_corrects / total

    train_acc, train_precision, train_recall, train_f1, train_roc_auc = calculate_metrics(
        np.array(all_train_labels), np.array(all_train_preds), np.array(all_train_probs)
    )

    # 保存训练集的指标
    train_losses.append(total_loss)
    train_accs.append(train_acc)
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    train_f1s.append(train_f1)
    train_roc_aucs.append(train_roc_auc)

    print(
        f"epoch: {epoch + 1} | 训练集 loss: {total_loss:.5f} | 训练集acc: {acc:.2f}% | precision: {train_precision:.4f} | 训练集 recall: {train_recall:.4f} | F1_score: {train_f1:.4f} | ROC AUC: {train_roc_auc:.4f}")

    return total_loss, acc, train_precision, train_recall, train_f1, train_roc_auc


# 验证函数
def validate_model(model, val_loader, loss_fn, optimizer, epoch):
    model.eval()
    total_loss = 0.
    total_corrects = 0.
    total = 0.
    all_val_labels = []
    all_val_preds = []
    all_val_probs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            preds = outputs.argmax(dim=1)
            probs = nn.Softmax(dim=1)(outputs).detach().cpu().numpy()  # 获取概率

            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(preds.cpu().numpy())
            all_val_probs.extend(probs)

            total_corrects += torch.sum(preds.eq(labels))
            total_loss += loss.item() * inputs.size(0)
            total += labels.size(0)

        total_loss = total_loss / total
        accuracy = 100 * total_corrects / total

        val_acc, val_precision, val_recall, val_f1, val_roc_auc = calculate_metrics(
            np.array(all_val_labels), np.array(all_val_preds), np.array(all_val_probs)
        )

        # 保存验证集的指标
        val_losses.append(total_loss)
        val_accs.append(val_acc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        val_roc_aucs.append(val_roc_auc)

        print(
            f"epoch: {epoch + 1} | 验证集 loss: {total_loss:.5f} | 验证集 acc: {accuracy:.2f}% | precision: {val_precision:.4f} | 验证集 recall: {val_recall:.4f} | F1_score: {val_f1:.4f} | ROC AUC: {val_roc_auc:.4f}")

    return total_loss, accuracy, val_precision, val_recall, val_f1, val_roc_auc


# 设置损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 训练和验证过程
for epoch in range(epoches):
    train_metrics = train_model(model, train_loader, loss_fn, optimizer, epoch)
    val_metrics = validate_model(model, val_loader, loss_fn, optimizer, epoch)
    save_metrics(epoch, train_metrics, val_metrics, "/media/disk/Backup/03drive/03SJY/samples/htr/128.txt")
