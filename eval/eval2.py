
import os
import sys
import argparse
import numpy as np
import csv
import torch
import point_cloud_utils as pcu

from glob import glob
from tqdm import tqdm

sys.path.append(os.getcwd())

from modules.utils.score_utils import chamfer_distance_unit_sphere
from modules.utils.score_utils import hausdorff_distance_unit_sphere
from modules.utils.score_utils import point_mesh_bidir_distance_single_unit_sphere


def evaluation(args):

    metric_global = { 'name': 'Total', 'CD': 0.0, 'P2S': 0.0, 'HD': 0.0 }
    if args.csv is not None:
        fcsv = open(args.csv, 'w', encoding='utf-8')
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow(['name', 'CD', 'P2S', 'HD'])

    pred_paths = glob(f'{args.pred_dir}/*.xyz')
    for pred_path in tqdm(pred_paths):
        if args.verbose:
            print(f'Evaluate {pred_path}...')

        _, file_name = os.path.split(pred_path)
        mesh_path = os.path.join(args.off_dir, file_name).replace('.xyz', '.off')
        gt_path   = os.path.join(args.gt_dir, file_name)

        pc_pred = torch.from_numpy(np.loadtxt(pred_path, dtype=np.float32))  # [N, 3]
        pc_gt   = torch.from_numpy(np.loadtxt(gt_path, dtype=np.float32))    # [N, 3]

        bpc_pred = pc_pred.view(1, pc_pred.shape[0], pc_pred.shape[1])
        bpc_gt   =   pc_gt.view(1,   pc_gt.shape[0],   pc_gt.shape[1])

        verts, faces = pcu.load_mesh_vf(mesh_path)
        verts = torch.from_numpy(verts.astype(np.float32))
        faces = torch.from_numpy(faces.astype(np.int64))

        # loss_cd = pytorch3d.loss.chamfer_distance(bpc_pred, bpc_gt)[0].item()
        loss_cd  = chamfer_distance_unit_sphere(bpc_pred, bpc_gt)[0].item()
        loss_hd  = hausdorff_distance_unit_sphere(bpc_pred, bpc_gt)[0].item()
        loss_p2f = point_mesh_bidir_distance_single_unit_sphere(pcl=pc_pred, verts=verts, faces=faces).item()

        metric_local = {}
        metric_local['name'] = file_name
        metric_local['CD']  = loss_cd
        metric_local['P2S'] = loss_p2f
        metric_local['HD']  = loss_hd
        if args.csv is not None:
            csv_writer.writerow(list(metric_local.values()))

        metric_global['CD']  += loss_cd
        metric_global['P2S'] += loss_p2f
        metric_global['HD']  += loss_hd
        
        # print(f'{file_name}: {loss_cd}')

        if args.verbose:
            print(metric_local)

    metric_global['CD']  = metric_global['CD']  / len(pred_paths)
    metric_global['P2S'] = metric_global['P2S'] / len(pred_paths)
    metric_global['HD']  = metric_global['HD']  / len(pred_paths)
    
    print(f'Evaluation: [CD]{metric_global["CD"]}, [P2S]{metric_global["P2S"]}, [HD]{metric_global["HD"]}')

    if args.csv is not None:
        csv_writer.writerow(list(metric_global.values()))
        fcsv.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to predict directory')
    parser.add_argument('--off_dir', type=str, required=True, help='Path to ground-truth mesh directory')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to ground-truth points directory')
    parser.add_argument('--csv', type=str, default=None, help='Path to save csv')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    evaluation(args)
