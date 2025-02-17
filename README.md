# DME
1.segmentation.py
  代码包含scse注意力机制
  对于五种不同病灶分割，需要改病灶数据集路径，骨干网络对比（此处为densenet161，其余可自行修改）
  输出最佳分割模型路径
2.no_attention.py
  该代码用于消融，去除了scse注意力机制，在病灶分割的最佳骨干网络下进行实验
3.albumentation.py
  该代码用于消融，增加了图像增强操作
4./cut
  该文件夹下有五类病灶，分别为棉絮斑、水肿、硬性渗出、血管、出血
  将DME数据集进行一系列病灶分割，输出对应病灶的掩码图像
5.combined.py
  将生成的五种病灶掩码合并
6./clustering
  该文件夹下包含了三种聚类方法，分别为Kmedoids、kmeans和Agglomerative
  三类聚类方法下分别有八种骨干网络
