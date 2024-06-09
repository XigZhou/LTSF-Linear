#  from mlp_mixer_pytorch import MLPMixer
#
# num_classes = len(class_names)  # 根据数据集的类别数量来设置模型的输出类别数量
#
# # 构建MLP-Mixer模型
# model = MLPMixer(
#     image_size=img_height,  # 图像的高和宽
#     channels=3,  # 图像的通道数
#     patch_size=16,  # MLP-Mixer的patch大小
#     dim=512,  # MLP-Mixer的维度
#     depth=12,  # MLP-Mixer的深度
#     num_classes=num_classes  # 输出类别数量
# )
#
# # 将模型移动到GPU
# model = model.to(device)
#
# # 打印模型摘要
# print(model)