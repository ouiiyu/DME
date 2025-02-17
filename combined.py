import os
import shutil
"改为现在要合并的代码！！！！！！！"
original_image_path = '/media/disk/Backup/03drive/02LMW/from02drive/htr_final/dataset/htr1/'
segmented_image_paths = [
    "/media/disk/Backup/03drive/02LMW/from02drive/htr_end/attention/hard/1/",
    "/media/disk/Backup/03drive/02LMW/from02drive/htr_end/attention/cotton/1/",
    "/media/disk/Backup/03drive/02LMW/from02drive/htr_end/attention/disc/1/",
    '/media/disk/Backup/03drive/02LMW/from02drive/htr_end/attention/he/1/',
    '/media/disk/Backup/03drive/02LMW/from02drive/htr_end/attention/mico/1/',
    '/media/disk/Backup/03drive/02LMW/from02drive/htr_end/attention/vessel/1/'
]

'''把这个路径换掉！！！！'''
output_path = "/media/disk/Backup/03drive/02LMW/from02drive/htr_end/attention/no_attent/1/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

for filename in os.listdir(original_image_path):
    if filename.endswith(".png") or filename.endswith(".jpeg"):  # 过滤出图像文件
        original_img_full_path = os.path.join(original_image_path, filename)
        image_folder_name = filename.split('.')[0]  # 使用原始文件名（不带扩展名）作为文件夹名
        combined_folder_path = os.path.join(output_path, image_folder_name)

        if not os.path.exists(combined_folder_path):
            os.makedirs(combined_folder_path)

        shutil.copy(original_img_full_path, os.path.join(combined_folder_path, "original_image.png"))
        print(f"复制了原始图像 {original_img_full_path} 到 {combined_folder_path}")

        # 对于每个分割文件夹，查找与原始图像对应的文件（假设分割图像以 .png 结尾）
        base_filename = filename.split('.')[0] 
        
        
        # 打印调试信息以验证文件名和路径
        #print(f"正在处理原始图像文件名: {filename}, 基础名: {base_filename}")

        for idx, segmented_path in enumerate(segmented_image_paths):
            segmented_img_filename = f"{base_filename}.png"
            segmented_img_full_path = os.path.join(segmented_path, segmented_img_filename)
            print(f"查找: {segmented_img_full_path}")

            if os.path.exists(segmented_img_full_path):
                shutil.copy(segmented_img_full_path, os.path.join(combined_folder_path, f"lesion_{idx + 1}.png"))
                print(f"复制了分割图像 {segmented_img_full_path} 到 {combined_folder_path}")
            else:
                print(f"未找到对应的分割图像: {segmented_img_full_path}")

print("ok")
