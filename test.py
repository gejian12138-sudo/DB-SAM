import argparse
import os
import datetime
import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import pandas as pd
from tqdm import tqdm

from dataset import PolypTestDataset, collate_fn
from models.db_sam import build_db_sam
from utils.utils import get_logger, setup_seeds
from utils.metrics import SegMetrics
from config import config_dict

ALL_DATASETS = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']


def format_memory(mem):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if mem < 1024.0:
            return f"{mem:.2f}{unit}"
        mem /= 1024.0
    return f"{mem:.2f}GB"


def get_memory_usage(device):
    if device.type == 'cuda':
        return {
            'current_alloc': torch.cuda.memory_allocated(device=device),
            'current_reserved': torch.cuda.memory_reserved(device=device),
            'peak_alloc': torch.cuda.max_memory_allocated(device=device),
            'peak_reserved': torch.cuda.max_memory_reserved(device=device),
        }
    return {}


def reset_memory_stats(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device=device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="DB_SAM", help="run name")
    parser.add_argument("--batch_size", type=int, default=10, help="test batch size")
    parser.add_argument("--metrics", nargs="+", default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda', help="inference device")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config_dict["work_dir"], "model_train/DB_SAM/DB_SAM_best.pth"),
                        help="model checkpoint path")
    parser.add_argument("--num_workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--data_root", type=str,
                        default='./dataset/data/test/TestDataset',
                        help="test dataset root")
    parser.add_argument("--datasets", nargs="+", default=None, choices=ALL_DATASETS,
                        help=f"datasets to test, default all: {ALL_DATASETS}")
    return parser.parse_args()


def to_device(samples, device):
    for sample in samples:
        for k, v in list(sample.items()):
            if isinstance(v, torch.Tensor):
                sample[k] = v.float().to(device)


def test_one_epoch(args, model, test_loader, logger, work_dir, metrics_list):
    model.eval()
    reset_memory_stats(torch.device(args.device))
    total_metrics = [0.0] * len(metrics_list)
    records = []
    total_samples = 0
    total_e2e_time = 0.0
    total_comp_time = 0.0
    peak_mem_allocated = 0
    peak_mem_reserved = 0

    results_dir = os.path.join(work_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    for batch_idx, samples in enumerate(test_loader):
        pre_mem = get_memory_usage(torch.device(args.device))
        start_e2e = time.time()
        to_device(samples, torch.device(args.device))

        if 'cuda' in args.device:
            start_comp = torch.cuda.Event(enable_timing=True)
            end_comp = torch.cuda.Event(enable_timing=True)
            start_comp.record()
        else:
            start_comp = time.time()

        try:
            with torch.no_grad():
                outputs = model.infer(list_input=samples)
            if outputs is None:
                logger.error(f"Batch {batch_idx + 1}: model returned None")
                continue
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1}: inference failed - {str(e)}")
            continue

        if 'cuda' in args.device:
            end_comp.record()
            torch.cuda.synchronize(args.device)
            batch_comp_time = start_comp.elapsed_time(end_comp)
        else:
            batch_comp_time = (time.time() - start_comp) * 1000

        total_comp_time += batch_comp_time
        batch_e2e_time = (time.time() - start_e2e) * 1000
        total_e2e_time += batch_e2e_time

        post_mem = get_memory_usage(torch.device(args.device))
        if 'peak_alloc' in post_mem:
            peak_mem_allocated = max(peak_mem_allocated, post_mem['peak_alloc'])
            peak_mem_reserved  = max(peak_mem_reserved, post_mem['peak_reserved'])

        valid_idxs = [i for i, out in enumerate(outputs)
                      if 'masks' in out and out['masks'].shape == samples[i]['labels'].shape]
        if not valid_idxs:
            logger.info(f"Batch {batch_idx + 1}: invalid masks")
            continue
        total_samples += len(valid_idxs)

        batch_metrics = [SegMetrics(outputs[i]['masks'], samples[i]['labels'], metrics_list)
                         for i in valid_idxs]
        summed = [sum(m[i] for m in batch_metrics) for i in range(len(metrics_list))]
        total_metrics = [total_metrics[i] + summed[i] for i in range(len(metrics_list))]

        for i, met in zip(valid_idxs, batch_metrics):
            image_path = samples[i]['image_path']
            name = os.path.splitext(os.path.basename(image_path))[0]
            mask = outputs[i]['masks'].detach().cpu()
            save_image(mask, os.path.join(results_dir, f"{name}_pred.png"))
            records.append({'filename': f"{name}_pred.png", metrics_list[0]: float(met[0])})

        if (batch_idx + 1) % 51 == 0 or batch_idx == 0:
            mem_msg = ""
            if 'current_alloc' in post_mem:
                mem_alloc = post_mem['current_alloc'] - pre_mem.get('current_alloc', 0)
                mem_msg = f" | VRAM alloc: {format_memory(mem_alloc)}"
            logger.info(f"Batch {batch_idx + 1}: e2e: {batch_e2e_time:.2f}ms | "
                        f"comp: {batch_comp_time:.2f}ms{mem_msg}")

    avg_e2e = total_e2e_time / max(1, len(test_loader))
    avg_comp = total_comp_time / max(1, len(test_loader))
    logger.info(f"\n===== Latency & Memory =====\n"
                f"Avg e2e: {avg_e2e:.2f}ms | Avg comp: {avg_comp:.2f}ms\n"
                f"Peak VRAM alloc: {format_memory(peak_mem_allocated)} | "
                f"reserved: {format_memory(peak_mem_reserved)}\n"
                f"==============================")

    return total_metrics, total_samples, records


def print_summary_table(all_results: dict, metrics_list: list, logger):
    datasets = list(all_results.keys())
    if not datasets:
        return
    metric_cols = [m.upper() for m in metrics_list]
    col_widths = [max(20, max(len(d) for d in datasets))] + [10] * len(metric_cols)
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_cells = ["Dataset".center(col_widths[0])] + [m.center(col_widths[i + 1]) for i, m in enumerate(metric_cols)]
    header = "|" + "|".join(f" {c} " for c in header_cells) + "|"
    lines = [sep, header, sep]
    for ds_name in datasets:
        vals = all_results[ds_name]
        cells = [ds_name.ljust(col_widths[0])]
        for i, v in enumerate(vals):
            cells.append(f"{v:.4f}".center(col_widths[i + 1]))
        lines.append("|" + "|".join(f" {c} " for c in cells) + "|")
    lines.append(sep)
    if len(datasets) > 1:
        mean_vals = [sum(all_results[d][i] for d in datasets) / len(datasets)
                     for i in range(len(metrics_list))]
        cells = ["Mean".ljust(col_widths[0])]
        for i, v in enumerate(mean_vals):
            cells.append(f"{v:.4f}".center(col_widths[i + 1]))
        lines.append("|" + "|".join(f" {c} " for c in cells) + "|")
        lines.append(sep)
    table_str = "\n".join(lines)
    print(table_str)
    logger.info(table_str)


def main():
    args = parse_args()
    setup_seeds()

    target_datasets = args.datasets if args.datasets else ALL_DATASETS
    datasets_to_test = [(ds, os.path.join(args.data_root, ds)) for ds in target_datasets
                        if os.path.isdir(os.path.join(args.data_root, ds))]
    if not datasets_to_test:
        print("[ERROR] No valid test datasets found.")
        return

    model = build_db_sam(checkpoint=args.checkpoint, load_ori_checkpoint=False).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e6:.2f}M")

    work_dir = os.path.join(config_dict['work_dir'], args.run_name)
    os.makedirs(work_dir, exist_ok=True)
    log_dir = os.path.join(work_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = get_logger(os.path.join(log_dir, f"{args.run_name}_{timestamp}.log"))
    logger.info(f"Args: {args}")

    all_results = {}
    for ds_name, ds_dir in datasets_to_test:
        print(f"\nTesting: {ds_name}")
        dataset = PolypTestDataset(data_dir=ds_dir, requires_name=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            shuffle=False, collate_fn=collate_fn)
        ds_work_dir = os.path.join(work_dir, ds_name)
        os.makedirs(ds_work_dir, exist_ok=True)
        metrics_sum, total_samples, recs = test_one_epoch(args, model, loader, logger, ds_work_dir, args.metrics)
        if total_samples > 0:
            avg_metrics = [metrics_sum[i] / total_samples for i in range(len(metrics_sum))]
        else:
            avg_metrics = [0.0] * len(metrics_sum)
        all_results[ds_name] = avg_metrics
        logger.info(f"[{ds_name}] {dict(zip(args.metrics, avg_metrics))}")

    print_summary_table(all_results, args.metrics, logger)

    summary_rows = []
    for ds_name, vals in all_results.items():
        row = {'Dataset': ds_name}
        row.update({m.upper(): v for m, v in zip(args.metrics, vals)})
        summary_rows.append(row)
    if len(all_results) > 1:
        mean_row = {'Dataset': 'Mean'}
        mean_row.update({m.upper(): sum(all_results[d][i] for d in all_results) / len(all_results)
                         for i, m in enumerate(args.metrics)})
        summary_rows.append(mean_row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(work_dir, f'summary_{timestamp}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
