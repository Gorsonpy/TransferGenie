import glob
import os
from pathlib import Path

import torch

def _load_checkpoint_for_ema(model_ema, checkpoint):
    # load ema model, maybe finish in future
    pass

def auto_load_model(args, model, optimizer, scheduler, loss_scaler, model_ema=None):
    out_dir = Path(args.save)
    if args.auto_resume and len(args.resume) == 0:
        all_checkpoints = glob.glob(os.path.join(out_dir, 'checkpoint-best.pth'))
        if len(all_checkpoints) > 0:
            args.resume = os.path.join(out_dir, 'checkpoint-best.pth')
        else:
            all_checkpoints = glob.glob(os.path.join(out_dir, "checkpoint-*.pth"))
            latest_idx = -1
            for ckpt in all_checkpoints:
                idx = ckpt.split('-')[-1].split('.')[0]
                if idx.isdigit():
                    latest_idx = max(int(idx), latest_idx)
            if latest_idx >= 0:
                args.resume = os.path.join(out_dir, 'checkpoint-{}.pth'.format(latest_idx))
        print("Auto resume checkpoint: ", args.resume)

    if args.resume:
        if args.resume.startswith('http'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print("Resume checkpoint from: ", args.resume)

            if 'epoch' in checkpoint:
                print("Resume checkpoint at epoch: ", checkpoint['epoch'])
                args.start_epoch = checkpoint['epoch'] + 1
            if 'optimizer' in checkpoint:
                print('Resume optimizer...')
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                print('Resume scheduler...')
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'scaler' in checkpoint:
                print('Resume scaler...')
                loss_scaler.load_state_dict(checkpoint['scaler'])




