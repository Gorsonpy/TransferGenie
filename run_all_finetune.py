import argparse
import os
import time
from pathlib import Path

import torch
from torchvision import models
from transformers import get_cosine_schedule_with_warmup

import utils
from data_loader.CUBLoader import create_cub_data_loader
from torch.cuda.amp import GradScaler
import torch.optim as optim

from engine_all_finetune import train_one_epoch, evaluate_one_epoch


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
    parser.add_argument('--clip_grad', type=float, default=None, help='gradient clipping')
    parser.add_argument('--model_ema', type=bool, default=False, help='whether to use model ema')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--resume', type=str, default='', help='path to resume checkpoint')
    parser.add_argument('--auto_resume', type=bool, default=True, help='whether to auto resume')
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument('--save', type=str, default='./checkpoints/cub_all_finetune', help='path to save checkpoints')
    return parser.parse_args()

def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("use device: ", device)
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)

    model = None
    dataset_path = None
    num_classes = 200
    if args.dataset == 'CUB':
        print("Loading pre-trained ResNet-50 model...")
        model = models.resnet50(weights='IMAGENET1K_V2')
        num_classes = 200
        dataset_path = '/Users/gorsonpy/dev/TransferGenie/dataset/CUB_200_2011/CUB_200_2011'

    # modify the last layer
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model: %s" % model)
    print("Number of trained parameters = %d" % n_parameters)
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
    loss_scaler = torch.cuda.amp.GradScaler()

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

