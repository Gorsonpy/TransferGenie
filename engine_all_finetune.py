import argparse
import time

import torch
from torch import autocast


def train_one_epoch(model, data_loader, criterion, optimizer, device, epoch, loss_scaler, clip_grad=None, arg=None):
    model.train()
    loss_total = 0.0
    acc_total = 0.0
    batch_count = 0
    sample_count = 0

    start_time = time.time()
    data_time_total = 0.0
    batch_time_total = 0.0

    for idx, (samples, labels) in enumerate(data_loader):
        batch_start = time.time()
        data_time = batch_start - (start_time + batch_time_total)
        data_time_total += data_time

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = samples.size(0)

        #
        with autocast():
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        loss_scaler.step(optimizer)
        loss_scaler.update()

        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_time_total += batch_time
