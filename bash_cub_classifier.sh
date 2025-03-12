nohup bash -c "python run_base_ft.py \
  --strategy only_classifier \
  --dataset CUB \
  --batch_size 512 \
  --num_workers 16 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-4 \
  --weight_decay 1e-4 " > cub_only_classifier.log 2>&1 &
echo "Process started with PID: $!"