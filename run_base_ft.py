import argparse
import os
import random
import time
from pathlib import Path

import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
from transformers import get_cosine_schedule_with_warmup

import utils
from data_loader.CUBLoader import create_cub_data_loader
from torch.cuda.amp import GradScaler
import torch.optim as optim

from engine_base_ft import train_one_epoch, evaluate_one_epoch
def get_args():
    parser = argparse.ArgumentParser(description='Fine-tune the whole model')
    parser.add_argument('--dataset', type=str, default='CUB', help='the name of dataset')

    # argument about training
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--pin_memory', type=bool, default=True, help='whether to pin memory')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

    # argument about optimizer（adamw）
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='betas')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps')

    # argument about scheduler
    parser.add_argument('--min_lr', type=float, default=1e-6, help='min learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs')

    # other
    parser.add_argument('--strategy', type=str, default='full',
                        choices=['full', 'only_classifier', 'from_scratch', 'last_layer',
                                 'last_2_layers', 'last_3_layers', 'last_4_layers', 'random_blocks'],
                        help='Fine-tuning strategy to use')
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')
    parser.add_argument('--clip_grad', type=float, default=None, help='gradient clipping')
    parser.add_argument('--model_ema', type=bool, default=False, help='whether to use model ema')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--resume', type=str, default='', help='path to resume checkpoint')
    parser.add_argument('--auto_resume', type=bool, default=True, help='whether to auto resume')
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument('--save', type=str, default='./checkpoints/cub_all_finetune', help='path to save checkpoints')
    return parser.parse_args()

def get_finetune_model(strategy, dataset):
    # dataset: the name of dataset
    num_classes = 0
    dataset_path = ''
    model = None
    if dataset == 'CUB':
        num_classes = 200
        dataset_path = Path('./dataset/CUB_200_2011/CUB_200_2011')
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if strategy != 'from_scratch':
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        print("Loading pre-trained ResNet-50 model...")
    else:
        model = models.resnet50(weights=None)
        print("Loading ResNet-50 model without pre-trained...")

    # modify the last layer
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    if strategy == 'full':
        print("Fine-tune the full model")
    elif strategy == 'only_classifier':
        print("Only fine-tune the classifier")
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    elif strategy == 'from_scratch':
        print("Train the model from scratch")
    elif strategy == 'last_layer':
        print("Fine-tune the last layer(layer4) and classifier")
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False

    elif strategy == 'last_2_layers':
        print("Fine-tune the last 2 layers(layer3, layer4) and classifier")
        for name, param in model.named_parameters():
            if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
    elif strategy == 'last_3_layers':
        print("Fine-tune the last 3 layers(layer2, layer3, layer4) and classifier")
        for name, param in model.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
    elif strategy == 'last_4_layers':
        print("Fine-tune the last 4 layers(layer1, layer2, layer3, layer4) and classifier")
        for name, param in model.named_parameters():
            if 'layer1' not in name and 'layer2' not in name and 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
    elif strategy == 'random_blocks':
        print("Fine-tune random blocks and classifier")
        # 随机微调50%的残差块
        # 首先冻结所有参数
        print("Randomly fine-tuning 50% of residual blocks")
        for param in model.parameters():
            param.requires_grad = False

        # 解冻分类头
        for param in model.fc.parameters():
            param.requires_grad = True

        # 获取所有块名称
        blocks = []
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, name)
            for i in range(len(layer)):
                blocks.append(f"{name}.{i}")

        # 随机选择50%的块进行微调
        num_blocks_to_finetune = len(blocks) // 2
        blocks_to_finetune = random.sample(blocks, num_blocks_to_finetune)
        print(f"Randomly selected blocks to fine-tune: {blocks_to_finetune}")

        # 解冻选定的块
        for name, param in model.named_parameters():
            for block in blocks_to_finetune:
                if block in name:
                    param.requires_grad = True
                    break
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    # 打印参与训练的参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    return model

def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("use device: ", device)
    args.save = f'./checkpoints/{args.dataset}_{args.strategy}'
    Path(args.save).mkdir(parents=True, exist_ok=True)

    model = get_finetune_model(args.strategy, args.dataset)
    model = model.to(device)
    print("Model: %s" % model)

    if args.dataset == 'CUB':
        dataset_path = Path('./dataset/CUB_200_2011/CUB_200_2011')
    print(f'loading {args.dataset} dataset from {dataset_path}')
    if args.dataset == 'CUB':
        train_loader = create_cub_data_loader(
            dataset_path=dataset_path,
            is_train=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        test_loader = create_cub_data_loader(
            dataset_path=dataset_path,
            is_train=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )

    print("Using Criterion: CrossEntropyLoss")
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Using AdamW optimizer with lr={args.lr}, weight_decay={args.weight_decay}, betas={args.betas}, eps={args.eps}")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=args.betas,
        eps=args.eps
    )

    print(f"Using get_cosine_schedule_with_warmup scheduler with warmup_epochs={args.warmup_epochs}, num_epochs={args.epochs}")
    steps_per_epoch = len(train_loader)
    total_steps = len(train_loader) * args.epochs
    warm_steps = args.warmup_epochs * steps_per_epoch
    print(f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warm_steps={warm_steps}")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=total_steps,
    )

    # loss_scaler
    loss_scaler = torch.amp.GradScaler('cuda')

    print("Batch size = %d" % args.batch_size)
    print("Number of workers = %d" % args.num_workers)

    utils.auto_load_model(args = args, model = model, optimizer = optimizer, scheduler=scheduler, loss_scaler = loss_scaler)
    print(f"Start training for {args.epochs} epochs")
    start_times = time.time()

    best_acc = 0.0
    patience_encounter = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            clip_grad=args.clip_grad,
            arg=args,
        )
        val_stats = evaluate_one_epoch(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        # early stop
        if val_stats['acc'] > best_acc:
            best_acc = val_stats['acc']
            best_path = Path(args.save) / 'checkpoint-best.pth'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict() if optimizer is not None else None,
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'epoch': epoch,
                'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
            }, best_path)
            print("Best checkpoint saved with accuracy: %.2f%%" % best_acc)
            patience_encounter = 0
        else:
            print(f"checkpoint is not the best, patience_encounter: {patience_encounter}")
            patience_encounter += 1

        if patience_encounter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    total_time = time.time() - start_times
    print(f"Total time: {total_time}")
    print("Training completed, Best accuracy: %.2f%%" % best_acc)

if __name__ == '__main__':
    opts = get_args()
    main(opts)

