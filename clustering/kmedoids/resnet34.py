import itertools
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def kmedoids(data, k, max_iter=300):
    """
    实现 K-Medoids 聚类算法
    
    参数：
    data: 输入的数据集，大小为 (n_samples, n_features)
    k: 聚类的数量
    max_iter: 最大迭代次数
    
    返回：
    cluster_labels: 聚类结果，每个样本对应的簇标签
    medoids: 每个簇的中心点
    """
    # 随机初始化 K 个中心点（medoids）
    n_samples = data.shape[0]
    initial_medoids_idx = np.random.choice(n_samples, size=k, replace=False)
    medoids = data[initial_medoids_idx]
    
    for iteration in range(max_iter):
        # 计算每个样本与所有中心点的距离
        distances = cdist(data, medoids, metric='euclidean')
        
        # 为每个样本分配最近的中心点
        cluster_labels = np.argmin(distances, axis=1)
        
        # 保存当前的 medoids，用于检查是否收敛
        old_medoids = medoids.copy()
        
        # 更新 medoids：对每个簇，选择簇内距离所有点最小的点作为新的 medoid
        for i in range(k):
            cluster_points = data[cluster_labels == i]
            if cluster_points.shape[0] > 0:
                dist_to_points = cdist(cluster_points, cluster_points, metric='euclidean')
                # 计算每个点与簇内其他点的距离之和
                medoid_idx = np.argmin(np.sum(dist_to_points, axis=1))
                medoids[i] = cluster_points[medoid_idx]
        
        # 如果 medoids 没有变化，表示收敛
        if np.all(medoids == old_medoids):
            break
    
    return cluster_labels, medoids
dataset_path_0 = '/media/disk/Backup/03drive/02LMW/from02drive/dme/cutmodel/mask/combined_/0'
dataset_path_1 = '/media/disk/Backup/03drive/02LMW/from02drive/dme/cutmodel/mask/combined_/1'
dataset_paths = [dataset_path_0, dataset_path_1]
label_ratios = [1, 1]  # 假设按1:1比例
dataset_labels = [0, 1]  # 只有两类

resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
resnet34 = nn.Sequential(*list(resnet34.children())[:-1])  # 去除最后的全连接层
resnet34.eval()
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
resnet34.to(DEVICE)




# 图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Resize((224, 224)),
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
output_dir_metrics = '/media/disk/Backup/03drive/02LMW/from02drive/dme/contrast/kmodes/resnet34_/weights_metrics'
output_dir_visualizations = '/media/disk/Backup/03drive/02LMW/from02drive/dme/contrast/kmodes/resnet34_/visual'

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
                    feature = resnet34(masked_image_tensor)
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

        # 选择聚合方式，例如求平均
        aggregated_feature = np.mean(lesion_features, axis=0)
        features.append(aggregated_feature)
        image_names.append(img_path)
        true_labels.append(image_labels_dict[img_path])

    # 对特征进行标准化和降维处理
    features_scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=50)  # 降维到 50 维
    features_scaled = pca.fit_transform(features_scaled)

    # 聚类
    k =2
    cluster_labels, medoids = kmedoids(features_scaled, k)

    # 计算聚类性能指标
    accuracy = accuracy_score(true_labels, cluster_labels)
    precision = precision_score(true_labels, cluster_labels, average='weighted')
    recall = recall_score(true_labels, cluster_labels, average='weighted')
    f1 = f1_score(true_labels, cluster_labels, average='weighted')
    silhouette = silhouette_score(features_scaled, cluster_labels)
    calinski = calinski_harabasz_score(features_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(features_scaled, cluster_labels)

    # 如果当前权重组合下的准确率更高，则保存该组合的聚类结果
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_weights = lesion_weights

    # 保存聚类结果和性能指标
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Silhouette Score": silhouette,
        "Calinski-Harabasz Score": calinski,
        "Davies-Bouldin Score": davies_bouldin
    }

    # 保存结果
    with open(os.path.join(output_dir_metrics, f"metrics_{file_counter}.txt"), 'w') as f:
        f.write(f"病灶权重组合: {lesion_weights}\n\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")

    # 可视化聚类结果
    plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=cluster_labels)
    plt.savefig(os.path.join(output_dir_visualizations, f"cluster_visualization_{file_counter}.png"))
    plt.close()

    file_counter += 1

# 输出最优权重组合和性能指标
print(f"最佳权重组合: {best_weights}")
