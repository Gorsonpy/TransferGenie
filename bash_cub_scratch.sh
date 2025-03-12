nohup bash -c "python run_base_ft.py \
  --strategy from_scratch \
  --dataset CUB \
  --batch_size 512 \
  --num_workers 16 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-4 \
  --weight_decay 1e-4 " > cub_from_scratch 2>&1 &
echo "Process started with PID: $!"