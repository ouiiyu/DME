import itertools
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


dataset_path_0 = '/media/disk/Backup/03drive/02LMW/from02drive/dme/cutmodel/mask/combined_/0'
dataset_path_1 = '/media/disk/Backup/03drive/02LMW/from02drive/dme/cutmodel/mask/combined_/1'
dataset_paths = [dataset_path_0, dataset_path_1]
label_ratios = [1, 1]  
dataset_labels = [0, 1]  # 只有两类

resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18 = nn.Sequential(*list(resnet18.children())[:-1])  # 去除最后的全连接层
resnet18.eval()
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
resnet18.to(DEVICE)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 初始化病灶权重
lesion_types = ["lesion_1", "lesion_2", "lesion_3", "lesion_4"]
weight_range = [0.1, 1.0, 10.0]  # 扩大权重范围
area_threshold = 10  # 病灶面积最小阈值
best_accuracy = 0
best_weights = None

# 输出文件和图像保存路径
output_dir_metrics = '/media/disk/Backup/03drive/02LMW/from02drive/dme/agg/resnet18__/weights_metrics'
output_dir_visualizations = '/media/disk/Backup/03drive/02LMW/from02drive/dme/agg/resnet18__/visual'

# 确保输出目录存在
os.makedirs(output_dir_metrics, exist_ok=True)
os.makedirs(output_dir_visualizations, exist_ok=True)

# 计数器用于生成唯一的文件名
file_counter = 1

# 网格搜索所有可能的权重组合
for weights in itertools.product(weight_range, repeat=len(lesion_types)):
    lesion_weights = dict(zip(lesion_types, weights))
    print(f"当前权重组合: {lesion_weights}")
    features = []
    image_names = []
    true_labels = []


    # 存储每个图像的病灶特征
    image_features_dict = {}  # {image_path: [weighted_feature1, weighted_feature2, ...]}
    image_labels_dict = {}    # {image_path: true_label}

    # 遍历每个眼底图像的文件夹
    for dataset_path, true_label, ratio in zip(dataset_paths, dataset_labels, label_ratios):
        all_image_folders = os.listdir(dataset_path)
        num_folders = len(all_image_folders)
        num_to_select = num_folders * ratio // sum(label_ratios)  # 按照比例选取数据

        selected_folders = all_image_folders[:num_to_select]

        for image_folder in selected_folders:
            image_folder_path = os.path.join(dataset_path, image_folder)
            if not os.path.isdir(image_folder_path):
                continue  # 跳过不是目录的项
            original_image_path = os.path.join(image_folder_path, 'original_image.png')  # 假设每个文件夹有原始图像
            original_image = cv2.imread(original_image_path)

            if original_image is None:
                print(f"警告: 无法读取原始图像 {original_image_path}，跳过该文件夹。")
                continue
            original_image = original_image.astype(np.float32) / 255.0

            # 初始化该图像的特征列表
            if original_image_path not in image_features_dict:
                image_features_dict[original_image_path] = []
                image_labels_dict[original_image_path] = true_label

            # 遍历每个分割后的病灶图像
            for lesion_file in os.listdir(image_folder_path):
                if lesion_file == 'original_image.png':
                    continue  # 跳过原始图像文件

                lesion_path = os.path.join(image_folder_path, lesion_file)

                # 读取病灶分割 mask 图像
                lesion_mask = cv2.imread(lesion_path, cv2.IMREAD_GRAYSCALE)
                if lesion_mask is None:
                    print(f"警告: 无法读取病灶图像 {lesion_path}，跳过该文件。")
                    continue

                # 调整原图大小以匹配 mask
                if original_image.shape[:2] != lesion_mask.shape[:2]:
                    print(f"尺寸不匹配，调整原图大小：原图尺寸 {original_image.shape[:2]} -> 掩码尺寸 {lesion_mask.shape[:2]}")
                    original_image_resized = cv2.resize(original_image, (lesion_mask.shape[1], lesion_mask.shape[0]))
                else:
                    original_image_resized = original_image

                # 获取病灶类型
                lesion_type = lesion_file.split('.')[0]  # 假设文件名如 'lesion_2.png'

                # 对 lesion_2 反转掩码
                if lesion_type == "lesion_3":
                    lesion_mask = cv2.bitwise_not(lesion_mask)

                # 计算病灶区域面积并筛选
                contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filtered_mask = np.zeros_like(lesion_mask)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > area_threshold:  # 仅保留面积大于阈值的区域
                        cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

                # 将筛选后的掩码调整为原图尺寸
                lesion_resized = cv2.resize(filtered_mask, (original_image_resized.shape[1], original_image_resized.shape[0]))

                # 将掩码应用到原始图像
                masked_image = cv2.bitwise_and(original_image_resized, original_image_resized, mask=lesion_resized)

                # 在传递到 transform 之前，将数据类型转换为 uint8
                masked_image_normalized = (masked_image * 255).astype(np.uint8)

                # 检查掩码是否为空
                if np.sum(lesion_resized) == 0:
                    print(f"警告: 病灶 {lesion_file} 的掩码为空，跳过该病灶。")
                    continue

                # 转换为 RGB 格式
                masked_image_rgb = cv2.cvtColor(masked_image_normalized, cv2.COLOR_BGR2RGB)

                # 传递给 transform
                masked_image_tensor = transform(masked_image_rgb)
                masked_image_tensor = masked_image_tensor.unsqueeze(0).to(DEVICE)

                # 提取特征
                with torch.no_grad():
                    feature= resnet18(masked_image_tensor)
                    feature = feature.flatten().cpu().numpy()

                # 获取当前病灶类型的权重
                weight = lesion_weights.get(lesion_type, 0)
                print(f"病灶类型: {lesion_type}, 权重: {weight}")  # 打印病灶类型和权重

                # 对特征进行加权
                weighted_feature = feature * weight

                # 将加权特征添加到对应图像的特征列表中
                image_features_dict[original_image_path].append(weighted_feature)

    # 对每个图像，聚合其所有病灶特征，得到图像级别的特征向量
    for img_path in image_features_dict:
        lesion_features = image_features_dict[img_path]
        if len(lesion_features) == 0:
            print(f"警告: 图像 {img_path} 没有有效的病灶特征，跳过该图像。")
            continue

        # 选择聚合方式，例如求和或平均
        aggregated_feature = np.sum(lesion_features, axis=0)  # 求和

        features.append(aggregated_feature)
        image_names.append(img_path)
        true_labels.append(image_labels_dict[img_path])

    features = np.array(features)

    if features.size == 0:
        print("错误: 未提取到任何特征")
        continue

    true_labels = np.array(true_labels)

    # 对特征进行标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 使用 PCA 降维
    pca = PCA(n_components=50)  # 降到50维，您可以调整此参数
    features_pca = pca.fit_transform(features_scaled)

    # 使用层次聚类
    clustering = AgglomerativeClustering(n_clusters=3)
    labels_clustering = clustering.fit_predict(features_pca)

    # 计算混淆矩阵
    confusion_mat = confusion_matrix(true_labels, labels_clustering)

    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(-confusion_mat)

    # 创建标签映射字典
    mapping = {col: row for row, col in zip(row_ind, col_ind)}

    # 重新映射聚类标签
    labels_mapped = np.array([mapping[label] for label in labels_clustering])

    # 计算评价指标
    accuracy = accuracy_score(true_labels, labels_mapped)
    precision = precision_score(true_labels, labels_mapped, average='macro')
    recall = recall_score(true_labels, labels_mapped, average='macro')
    f1 = f1_score(true_labels, labels_mapped, average='macro')
    silhouette = silhouette_score(features_pca, labels_clustering)
    calinski_harabasz = calinski_harabasz_score(features_pca, labels_clustering)
    davies_bouldin = davies_bouldin_score(features_pca, labels_clustering)

    print(f"当前准确率: {accuracy * 100:.2f}%")

    # 保存当前权重组合和评价指标
    output_txt_path = os.path.join(output_dir_metrics, f'weights_metrics_{file_counter}.txt')
    with open(output_txt_path, 'w') as f:
        f.write(f"当前权重组合: {lesion_weights}\n")
        f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
        f.write(f"精确率 (Precision): {precision:.4f}\n")
        f.write(f"召回率 (Recall): {recall:.4f}\n")
        f.write(f"F1 值 (F1 Score): {f1:.4f}\n")
        f.write(f"轮廓系数（Silhouette Coefficient）: {silhouette:.4f}\n")
        f.write(f"Calinski-Harabasz 指数: {calinski_harabasz:.4f}\n")
        f.write(f"Davies-Bouldin 指数: {davies_bouldin:.4f}\n")

    file_counter += 1

    # 更新最佳权重组合
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_weights = weights

# 输出最佳结果
print(f"最佳权重组合: {dict(zip(lesion_types, best_weights))}")
print(f"最佳聚类准确率为: {best_accuracy * 100:.2f}%")