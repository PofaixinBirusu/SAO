import numpy as np
import open3d as o3d
import torch
from torchvision.transforms import Compose
from modelnet40.datasets import ModelNetHdf
from modelnet40.datasets import get_transforms
from pre_encode.lrf import LRF
from pre_encode.ball_surface import compute
from copy import deepcopy
from common.math import se3
from torch.utils import data


def get_pc(pts, normal=None, color=None):
    pc = o3d.PointCloud()
    pc.points = o3d.Vector3dVector(pts)
    if normal is not None:
        pc.normals = o3d.Vector3dVector(normal)
    if color is not None:
        pc.colors = o3d.Vector3dVector([color]*pts.shape[0])
    return pc


class Modelnet40Pair(data.Dataset):
    def __init__(self, path, r_lrf, theta_space, mode="train", pre_encode_path=None, noise_type="crop"):
        super(Modelnet40Pair, self).__init__()
        train_transforms, test_transforms = get_transforms(noise_type=noise_type)
        train_transforms, test_transforms = Compose(train_transforms), Compose(test_transforms)

        tr = train_transforms if mode == "train" else test_transforms
        self.data = ModelNetHdf(path, subset=mode, transform=tr, categories=None)
        self.r_lrf = r_lrf
        self.space = theta_space
        self.pre_encode_path = pre_encode_path

    def __len__(self):
        # return 100
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        source_pts, source_normal = x["points_src"][:, :3], x["points_src"][:, 3:]
        target_pts, target_normal = x["points_ref"][:, :3], x["points_ref"][:, 3:]
        raw_pts = x["points_raw"][:, :3]
        idx = x["idx"]
        T = np.concatenate([x["transform_gt"], np.array([[0, 0, 0, 1]])], axis=0)
        source_pc = get_pc(source_pts, source_normal, [1, 0.706, 0])
        target_pc = get_pc(target_pts, target_normal, [0, 0.651, 0.929])
        # 找重叠
        source_tree, target_tree = o3d.KDTreeFlann(source_pc), o3d.KDTreeFlann(target_pc)

        match = np.zeros((source_pts.shape[0], target_pts.shape[0]))
        for i, pt in enumerate(np.asarray(deepcopy(source_pc).transform(T).points)):
            _, inds, _ = target_tree.search_radius_vector_3d(pt, 0.06)
            inds = np.asarray(inds)
            if inds.shape[0] > 1:
                match[i, inds] = 1
        source_lrf, target_lrf = LRF(source_pc, source_tree, self.r_lrf), LRF(target_pc, target_tree, self.r_lrf)
        source_key_feature, target_key_feature = [], []
        if self.pre_encode_path is None:
            for pt in source_pts:
                patch = source_lrf.get(pt)
                desc = compute(patch, self.space, self.space)
                desc[np.isnan(desc)] = 0
                source_key_feature.append(desc)
            source_pre_encode = np.stack(source_key_feature, axis=0)
            for pt in target_pts:
                patch = target_lrf.get(pt)
                desc = compute(patch, self.space, self.space)
                desc[np.isnan(desc)] = 0
                target_key_feature.append(desc)
            target_pre_encode = np.stack(target_key_feature, axis=0)
        else:
            pre_encode = np.load(self.pre_encode_path+"/"+"%d.npy" % index)
            source_pre_encode, target_pre_encode = pre_encode[:source_pts.shape[0], :], pre_encode[source_pts.shape[0]:, :]

        return source_pts, source_normal, source_pre_encode, target_pts, target_normal, target_pre_encode, match, raw_pts, T, idx