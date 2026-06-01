import argparse
import torch
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt

from torch import optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import PolypTrainDataset, collate_fn
from models.db_sam import DB_SAM
from utils.loss import FocalDiceloss
from utils.utils import get_logger, setup_seeds
from utils.metrics import SegMetrics
from config import config_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="DB_SAM", help="run model name")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=3, help="train batch size")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume")
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="gradient accumulation steps")
    return parser.parse_args()


def to_device(batch_input, device):
    device_input = []
    for one_dict in batch_input:
        one_dict_device = {
            key: (value.float().to(device) if key in ['image', 'labels', 'boxes'] else value)
            for key, value in one_dict.items()}
        device_input.append(one_dict_device)
    return device_input


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, loggers, scaler):
    model.train()
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    train_losses, train_focal_losses, train_dice_losses = [], [], []
    train_iter_metrics = [0] * len(args.metrics)
    grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
    optimizer.zero_grad()

    for batch, dict_list in enumerate(train_loader_tqdm):
        dict_list = to_device(dict_list, args.device)
        with autocast('cuda'):
            out = model.forward(list_input=dict_list)
            masks_cat = torch.cat([d['masks'] for d in out], dim=0)
            labels_cat = torch.cat([d['labels'] for d in dict_list], dim=0)
            focal_loss, dice_loss = criterion(masks_cat, labels_cat)
            loss = focal_loss + 20 * dice_loss
            loss_scaled = loss / grad_accum_steps

        scaler.scale(loss_scaled).backward()

        if (batch + 1) % grad_accum_steps == 0 or (batch + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_size = len(dict_list)
        train_batch_metrics = sum([SegMetrics(out[i]["masks"], dict_list[i]["labels"], args.metrics)
                                   for i in range(batch_size)]) / batch_size
        if int(batch + 1) % 725 == 0 or batch == 0:
            loggers.info(f'Epoch: {epoch + 1}, Batch: {batch + 1}, Loss: {loss.item():.4f}, '
                         f'Focal: {focal_loss.item():.4f}, Dice: {dice_loss.item():.4f}, '
                         f'Metrics: {train_batch_metrics}')
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]
        train_losses.append(loss.item())
        train_focal_losses.append(focal_loss.item())
        train_dice_losses.append(dice_loss.item())
        train_loader_tqdm.set_postfix(loss=f'{loss.item():.4f}')

    return train_losses, train_focal_losses, train_dice_losses, train_iter_metrics


def plot_and_save_losses(total_losses, focal_losses, dice_losses, run_name, output_dir):
    epochs = range(1, len(total_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_losses, 'r-', label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{run_name}_total_loss.png"))
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, focal_losses, 'b-', label='Focal Loss')
    plt.plot(epochs, dice_losses, 'g-', label='Dice Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Focal / Dice Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{run_name}_focal_dice_loss.png"))
    plt.close()


def main(args):
    setup_seeds()
    model = DB_SAM().to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = FocalDiceloss().to(args.device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler('cuda')

    if args.resume:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f, weights_only=False)
        model_state_dict = model.state_dict()
        pretrained_state_dict = {}
        for name, param in checkpoint['model'].items():
            if name in model_state_dict and model_state_dict[name].shape == param.shape:
                pretrained_state_dict[name] = param
            else:
                print(f"Skipping parameter {name} due to size mismatch.")
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict)
        print(f"Loaded checkpoint {args.resume}")

    train_dataset = PolypTrainDataset(data_dir=config_dict["train_dir"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True, collate_fn=collate_fn)

    log_file = os.path.join(config_dict["work_dir"], "logs",
                            args.run_name + f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}")
    loggers = get_logger(log_file)
    loggers.info(f"****{args}****")

    best_loss = float('inf')
    early_stop_patience = 20
    early_stop_counter = 0
    l_train = len(train_loader)
    epoch_total_losses, epoch_focal_losses, epoch_dice_losses = [], [], []

    for epoch in range(args.epochs):
        os.makedirs(os.path.join(config_dict["work_dir"], "model_train", args.run_name), exist_ok=True)
        train_losses, train_focal_losses, train_dice_losses, train_iter_metrics = train_one_epoch(
            args, model, optimizer, train_loader, epoch, criterion, loggers, scaler)
        avg_loss = np.mean(train_losses)
        avg_focal = np.mean(train_focal_losses)
        avg_dice = np.mean(train_dice_losses)
        scheduler.step(avg_loss)
        epoch_total_losses.append(avg_loss)
        epoch_focal_losses.append(avg_focal)
        epoch_dice_losses.append(avg_dice)
        train_iter_metrics = [metric / l_train for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}
        loggers.info(f"Train: epoch: {epoch + 1}, Train loss: {avg_loss:.4f}, "
                     f"Focal loss: {avg_focal:.4f}, Dice loss: {avg_dice:.4f}, "
                     f"Metrics: {train_metrics}")

        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   os.path.join(config_dict["work_dir"], "model_train", args.run_name, "DB_SAM_latest.pth"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0
            loggers.info(f"****BEST epoch: {epoch + 1}, Train loss: {avg_loss:.4f}****")
            torch.save({'model': model.float().state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(config_dict["work_dir"], "model_train", args.run_name, "DB_SAM_best.pth"))
        else:
            early_stop_counter += 1
            loggers.info(f"No improvement for {early_stop_counter} epochs (best_loss={best_loss:.4f}).")
            if early_stop_counter >= early_stop_patience:
                loggers.info("Early stopping triggered.")
                break

        plot_and_save_losses(epoch_total_losses, epoch_focal_losses, epoch_dice_losses,
                             args.run_name, config_dict["work_dir"])


if __name__ == '__main__':
    args = parse_args()
    main(args)
