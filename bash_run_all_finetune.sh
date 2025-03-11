nohup python run_all_finetune.py \
--dataset CUB \
--batch_size 32 \
--num_workers 8 \
--epochs 20 \
--patience 5 \
--lr 1e-4 \
--weight_decay 1e-4 \
--save ./checkpoints/cub_all_finetune
> cub_all_finetune.log 2>&1 &