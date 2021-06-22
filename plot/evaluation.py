
import os
import sys
import argparse
import numpy as np
import torch
import csv

sys.path.append(os.getcwd())

from metric.loss import ChamferDistance
from glob import glob


@torch.no_grad()
def evaluation(args):

    metric_global = { 'name': 'Total', 'CD': 0.0, 'P2S': 0.0 }
    if args.csv is not None:
        fcsv = open(args.csv, 'w', encoding='utf-8')
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow(['name', 'CD', 'P2S'])

    chamerLoss = ChamferDistance(dim=3)
    
    pred_paths = glob(f'{args.pred_dir}/*.xyz')
    for path in pred_paths:
        # if args.verbose:
        #     print(f'Evaluate {path}...')
        
        _, file_name = os.path.split(path)
        gt_path = os.path.join(args.gt_dir, file_name)

        pc_pred = np.loadtxt(path, dtype=np.float32)   # [N, 3]
        pc_pred = torch.from_numpy(pc_pred).unsqueeze(0).cuda()
        pc_gt = np.loadtxt(gt_path, dtype=np.float32)  # [N, 3]
        pc_gt = torch.from_numpy(pc_gt).unsqueeze(0).cuda()

        metric_local = {}
        metric_local['name'] = file_name
        metric_local['CD'] = chamerLoss(pc_pred, pc_gt).detach().cpu().item()
        metric_local['P2S'] = 0.0  # TODO:
        if args.csv is not None:
            csv_writer.writerow(list(metric_local.values()))

        metric_global['CD']  += metric_local['CD']
        metric_global['P2S'] += metric_local['P2S']

        if args.verbose:
            print(metric_local)

    metric_global['CD']  = metric_global['CD']  / len(pred_paths)
    metric_global['P2S'] = metric_global['P2S'] / len(pred_paths)
    print(f'Evaluation: [CD]{metric_global["CD"]}, [P2S]{metric_global["P2S"]}')

    if args.csv is not None:
        csv_writer.writerow(list(metric_global.values()))
        fcsv.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to predict directory')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to ground-truth directory')
    parser.add_argument('--csv', type=str, default=None, help='Path to save csv')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    evaluation(args)
