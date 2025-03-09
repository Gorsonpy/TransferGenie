from data_loader.CUBLoader import create_cub_data_loader

# 设置数据集路径
dataset_path = "/dataset/CUB_200_2011/CUB_200_2011"  # 修改为您的CUB数据集路径

if __name__ == '__main__':
    # 创建数据加载器（使用较小的batch_size进行测试）
    train_loader = create_cub_data_loader(
        dataset_path=dataset_path,
        is_train=True,
        batch_size=4,
        num_workers=2,
        pin_memory=True
    )

    # 获取一批数据并打印信息
    for images, labels in train_loader:
        print(f"Batch shapes: {images.shape}, {labels.shape}")
        print(f"Labels: {labels}")
        print(f"Image tensor min/max values: {images.min():.3f}/{images.max():.3f}")
        break  # 只检查第一批数据

    # 检查数据集大小
    train_dataset = train_loader.dataset
    test_loader = create_cub_data_loader(
        dataset_path=dataset_path,
        is_train=False,
        batch_size=4,
        num_workers=2,
        pin_memory=True
    )
    test_dataset = test_loader.dataset

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")