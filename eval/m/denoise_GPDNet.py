
import argparse
import numpy as np
import os
import sys

sys.path.append('/workspace/Denoise/GPDNet/Code/GPDNet_mse_sp')

from tqdm import tqdm
import pyHilbertSort as hilbersort
import point_cloud_utils as pcu

from Config import Config
from net_test_conv import Net
# from C2C_distance import compute_C2C
from knn_matrix import knn_matrix_from_data


MODEL_PATH1 = '/workspace/Denoise/GPDNet/Results/GPDNet_mse_sp/0.01/16nn/saved_models/'
MODEL_PATH2 = '/workspace/Denoise/GPDNet/Results/GPDNet_mse_sp/0.015/16nn/saved_models/'
MODEL_PATH3 = '/workspace/Denoise/GPDNet/Results/GPDNet_mse_sp/0.02/16nn/saved_models/'


parser = argparse.ArgumentParser()

# parser.add_argument('--model', default='DenoisePointNet.py', help='Model name: net')
parser.add_argument('--denoised_dir', default='', help='Testing results data directory')
parser.add_argument('--gt_dir', default='./Dataset/Test_Shapenet_h5/gt/', help='Testing gt data directory')
parser.add_argument('--noisy_dir', default='./Dataset/Test_Shapenet_h5/noisy/', help='Testing noisy data directory')
parser.add_argument('--model', default=1, type=int)


args = parser.parse_args()

if args.model == 1:
    RUN_MODEL_PATH = MODEL_PATH1
if args.model == 2:
    RUN_MODEL_PATH = MODEL_PATH2
if args.model == 3:
    RUN_MODEL_PATH = MODEL_PATH3

config1 = Config()
config1.save_dir = RUN_MODEL_PATH
model1 = Net(config1)
model1.do_variables_init()
model1.restore_model(config1.save_dir + 'model.ckpt')

# config2 = Config()
# config2.save_dir = MODEL_PATH2
# model2 = Net(config2)
# model2.do_variables_init()
# model2.restore_model(config2.save_dir + 'model.ckpt')

# config3 = Config()
# config3.save_dir = MODEL_PATH3
# model3 = Net(config3)
# model3.do_variables_init()
# model3.restore_model(config3.save_dir + 'model.ckpt')


class BatchIdxIter:
    def __init__(self, batch_size, N):
        self.batch_size = batch_size
        self.N = N
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        
        if self.current >= self.N:
            raise StopIteration

        if self.current + self.batch_size < self.N:
            bs = self.batch_size
        else:
            bs = self.N - self.current
        _range = list(range(self.current, self.current + bs))

        self.current += bs
        return bs, _range


def fps_numpy(pts, K):
    def calc_distances(p0, points):
        return ((p0 - points)**2).sum(axis=1)

    farthest_pts = np.zeros((K, 3), dtype=np.float32)
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)

    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts

def denoise_in_splitting_batch(model, noisy_pt, config, npoints, batch_size=16):
    
    denoised_patches = []
    for _, batch_idx in BatchIdxIter(batch_size, N=noisy_pt.shape[0]):
        noisy_batch = noisy_pt[batch_idx]

        # print("Computing knn_matrix")
        nn_matrix = knn_matrix_from_data(noisy_batch.reshape(-1, 3), config.knn)

        print(noisy_batch.shape, noisy_batch.shape[0] * noisy_batch.shape[1])
        denoise_batch = model.denoise(noisy_batch, nn_matrix, noisy_batch.shape[0] * noisy_batch.shape[1])
        denoised_patches.append(denoise_batch)
    denoised = np.concatenate(denoised_patches, axis=0)
    return denoised

def evaluate(source_path):

    _, filename = os.path.split(source_path)
    gt_path = '%s/%s' % (args.gt_dir, filename)
    target_path = '%s/%s' % (args.denoised_dir, filename)

    noisy_pt = np.loadtxt(source_path, dtype=np.float32)  # [N, 3]
    gt_pt    = np.loadtxt(gt_path,     dtype=np.float32)  # [N, 3]

    noisy_pt, _idx_sort = hilbersort.hilbertSort(3, noisy_pt)
    noisy_pt = noisy_pt.astype(np.float32)
    
    num_raw_point = noisy_pt.shape[0]

    # npoints = (noisy_pt.shape[0] / 1024) * 1024
    if num_raw_point == 10000:
        npoints = 1024 * 10
    elif num_raw_point == 50000:
        npoints = 1024 * 45
    else:
        assert False, 'Invalid number of points (only 10K or 50K point are support)'
    noisy_pt = fps_numpy(noisy_pt, npoints)
    noisy_pt = np.reshape(noisy_pt, [-1, 1024, 3])

    # denoise_1 = model1.denoise(noisy_pt, nn_matrix1, npoints)
    # denoise_2 = model2.denoise(noisy_pt, nn_matrix2, npoints)
    # denoise_3 = model3.denoise(noisy_pt, nn_matrix3, npoints)

    denoise_1 = denoise_in_splitting_batch(model1, noisy_pt, config1, npoints, batch_size=15)

    denoise_1 = np.reshape(denoise_1, [-1, 3])
    # denoise_2 = np.reshape(denoise_2, [-1, 3])
    # denoise_3 = np.reshape(denoise_3, [-1, 3])
    
    denoise_1 = fps_numpy(denoise_1, num_raw_point)

    C2C_noisy_0 = pcu.chamfer_distance(gt_pt, noisy_pt.reshape(-1, 3))
    C2C_noisy_1 = pcu.chamfer_distance(gt_pt, denoise_1)
    # C2C_noisy_2 = compute_C2C(gt_pt, denoise_2) * 1e+06
    # C2C_noisy_3 = compute_C2C(gt_pt, denoise_3) * 1e+06

    np.savetxt(target_path, denoise_1, fmt='%.6f')
    print('%s: %s -> %s' % (filename, C2C_noisy_0, C2C_noisy_1))

    # if C2C_noisy_1 < C2C_noisy_2 and C2C_noisy_1 < C2C_noisy_3:
    #     np.savetxt(denoise_1, target_path, fmt='%.6f')
    # elif C2C_noisy_2 < C2C_noisy_1 and C2C_noisy_2 < C2C_noisy_3:
    #     np.savetxt(denoise_2, target_path, fmt='%.6f')
    # else: # C2C_noisy_3 < C2C_noisy_1 and C2C_noisy_3 < C2C_noisy_2:
    #     np.savetxt(denoise_3, target_path, fmt='%.6f')
    
    

def mp_walkFile(directory):
    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, f) for f in files if '.xyz' in f]
        for path in tqdm(paths):
            evaluate(path)

mp_walkFile(args.noisy_dir)
