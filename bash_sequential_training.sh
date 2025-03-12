#!/bin/bash

# 创建日志目录
mkdir -p logs

# 定义默认数据集列表（可以根据需要修改）
datasets=("CUB")
# 如果需要多个数据集，可以取消下面的注释并添加更多数据集
# datasets=("CUB" "ImageNet" "CIFAR100")

# 定义要运行的策略列表
strategies=("full" "last_4_layers" "last_3_layers" "last_2_layers" "last_layer" "random_blocks" "only_classifier" "from_scratch")

echo "Starting sequential training for all strategies and datasets"

# 外层循环遍历数据集
for dataset in "${datasets[@]}"; do
    echo "===================================================="
    echo "Starting training on dataset: $dataset"
    echo "===================================================="

    # 内层循环遍历所有策略
    for strategy in "${strategies[@]}"; do
        echo "----------------------------------------"
        echo "Starting training for dataset '$dataset' with strategy: $strategy"
        echo "----------------------------------------"

        # 构建日志文件名（包含数据集和策略）
        log_file="logs/${dataset,,}_${strategy}.log"  # ${dataset,,} 将数据集名称转为小写

        # 启动Python脚本，但不使用nohup和&，这样会等待它完成
        echo "Running: python run_base_ft.py --strategy $strategy --dataset $dataset --batch_size 512 --num_workers 4 --epochs 100 --patience 10 --lr 1e-4 --weight_decay 1e-4"
        echo "Log file: $log_file"

        # 执行命令并同时输出到控制台和日志文件
        python run_base_ft.py \
          --strategy $strategy \
          --dataset $dataset \
          --batch_size 512 \
          --num_workers 4 \
          --epochs 100 \
          --patience 10 \
          --lr 1e-4 \
          --weight_decay 1e-4 2>&1 | tee $log_file

        # 检查上一个命令的退出状态
        if [ $? -eq 0 ]; then
            echo "Training for dataset '$dataset' with strategy '$strategy' completed successfully."
        else
            echo "WARNING: Training for dataset '$dataset' with strategy '$strategy' failed with exit code $?."
            echo "Do you want to continue with the next training? (y/n)"
            read answer
            if [[ $answer != "y" ]]; then
                echo "Exiting..."
                exit 1
            fi
        fi

        echo "Waiting 10 seconds before starting next training..."
        sleep 10
    done
done

echo "All training strategies and datasets completed!"