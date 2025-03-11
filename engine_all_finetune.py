import datetime
import time
from pathlib import Path

import torch
from torch import autocast


def train_one_epoch(model:torch.nn.Module, data_loader, criterion, optimizer, scheduler, device, epoch, loss_scaler, clip_grad=None, arg=None):
    model.train()
    loss_total = 0.0
    acc_total = 0.0
    batch_count = 0
    sample_count = 0

    grad_norm_total = 0.0
    max_grad_total = 0.0

    start_time = time.time()
    # data_time: load and preprocess data time
    data_time_total = 0.0
    # batch_time: process one batch total time
    batch_time_total = 0.0

    for idx, (samples, labels) in enumerate(data_loader):
        batch_start = time.time()
        data_time = batch_start - (start_time + batch_time_total)
        data_time_total += data_time

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = samples.size(0)

        #
        with autocast(device_type=device.type):
            outputs = model(samples)
            loss = criterion(outputs, labels)

        loss_value = loss.item()
        loss_total += loss_value * batch_size

        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        acc = correct / batch_size
        acc_total += correct

        batch_count += 1
        sample_count += batch_size

        optimizer.zero_grad()
        loss_scaler.scale(loss).backward()
        if clip_grad is not None:
            loss_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            max_grad = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_max_grad = p.grad.abs().max().item()
                    max_grad = max(param_max_grad, max_grad)
            grad_norm_total += grad_norm
            max_grad_total += max_grad

        else:
            grad_norm = 0.0
            max_grad = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.norm(2).item()
                    grad_norm += param_norm * param_norm
                    param_max = p.grad.abs().max().item()
                    max_grad = max(param_max, max_grad)
            grad_norm = grad_norm ** 0.5
            grad_norm_total += grad_norm
            max_grad_total += max_grad


        loss_scaler.step(optimizer)
        loss_scaler.update()

        if scheduler is not None:
            scheduler.step()

        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_time_total += batch_time

        if idx % 5 == 0 or idx == len(data_loader) - 1:
            avg_loss = loss_total / sample_count if sample_count > 0 else 0.0
            avg_acc = acc_total / sample_count * 100 if sample_count > 0 else 0.0
            avg_data_time = data_time_total / (idx + 1)
            avg_batch_time = batch_time_total / (idx + 1)
            samples_per_sec = batch_size / batch_time if batch_time > 0 else 0.0

            avg_grad_norm = grad_norm_total / (idx + 1)
            avg_max_grad = max_grad_total / (idx + 1)

            # estimate remaining time
            time_per_batch = batch_time_total / (idx + 1)
            eta_seconds = time_per_batch * (len(data_loader) - idx - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

            print(
                f"Epoch:[{epoch:>3d}/{arg.epochs}]  "
                f"Train: [{idx:>4d}/{len(data_loader)}]  "
                f"ETA: {eta_str}  "
                f"Grad: {grad_norm:.3e}({avg_grad_norm:.3e})  "
                f"Max Grad: {max_grad:.3e}({avg_max_grad:.3e})  "
                f"Loss: {loss_value:.4f}({avg_loss:.4f})  "
                f"Acc: {acc*100:.2f}%({avg_acc:.2f}%)  "
                f"LR: {optimizer.param_groups[0]['lr']:.3e}  "
                f"Data: {data_time:.3f}s({avg_data_time:.3f}s)  "
                f"Batch: {batch_time:.3f}s({avg_batch_time:.3f}s)  "
                f"Throughput: {samples_per_sec:.1f} samples/s"
            )
    avg_loss = loss_total / sample_count if sample_count > 0 else 0
    avg_acc = acc_total / sample_count * 100 if sample_count > 0 else 0
    avg_data_time = data_time_total / batch_count if batch_count > 0 else 0
    avg_batch_time = batch_time_total / batch_count if batch_count > 0 else 0
    avg_grad_norm = grad_norm_total / batch_count if batch_count > 0 else 0
    avg_max_grad = max_grad_total / batch_count if batch_count > 0 else 0

    epoch_time = time.time() - start_time
    print(f"Epoch [{epoch}] completed: Time: {epoch_time:.2f}s  Avg Loss: {avg_loss:.4f}  Avg Acc: {avg_acc:.2f}%  Avg Grad: {avg_grad_norm:.3e}  Avg Max Grad: {avg_max_grad:.3e}")

    train_stats = {
        'epoch': epoch,
        'loss': avg_loss,
        'acc': avg_acc,
        'lr': optimizer.param_groups[0]['lr'],
        'batch_time': avg_batch_time,
        'data_time': avg_data_time,
        'throughput': batch_size / avg_batch_time if avg_batch_time > 0 else 0,
        'grad_norm': avg_grad_norm,
        'max_grad': avg_max_grad,
    }

    save_checkpoint = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict() if optimizer is not None else None,
        'scheduler' : scheduler.state_dict() if scheduler is not None else None,
        'epoch' : epoch,
        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
    }

    save_path = Path(arg.save) / f'checkpoint-{epoch}.pth'
    torch.save(save_checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")
    return train_stats

def evaluate_one_epoch(model:torch.nn.Module, data_loader, criterion, device, epoch):
    model.eval()
    correct = 0.0
    total = 0.0
    val_loss = 0.0

    with torch.no_grad():
        for samples, labels in data_loader:
            samples, labels = samples.to(device), labels.to(device)
            outputs = model(samples)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * samples.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / total
    val_acc = 100.0 * correct / total
    print(f"Epoch [{epoch}] validation completed: Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")
    return {
        'loss':val_loss,
        'acc':val_acc,
    }

