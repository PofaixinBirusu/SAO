import numpy as np
import open3d as o3d
import torch
from copy import deepcopy
from common.math import se3
from dataset import Modelnet40Pair, get_pc
import utils
from utils import pc_normalize, ransac_pose_estimation
from model import SAOLoss, SAO, device


r_lrf = 0.4
theta_space = 1 / 6

pre_encode_path = "./modelnet_pre_encode/test"
dataset_path = "E:/modelnet40_ply_hdf5_2048"


def processbar(current, totle):
    process_str = ""
    for i in range(int(20*current/totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|  %d / %d" % (process_str, current, totle)


def chamfer_distance(source_pts, target_pts, raw_pts, T_pred, T_gt):
    def square_distance(src, dst):
        src, dst = torch.Tensor(src), torch.Tensor(dst)
        return torch.sum((src[:, None, :] - dst[None, :, :]) ** 2, dim=-1)

    source_pc = o3d.PointCloud()
    source_pc.points = o3d.Vector3dVector(source_pts)

    src_transformed = np.asarray(deepcopy(source_pc).transform(T_pred).points)
    ref_clean = raw_pts
    src_clean = se3.transform(se3.concatenate(T_pred[:3, :], se3.inverse(T_gt[:3, :])), raw_pts)
    dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
    dist_ref = torch.min(square_distance(target_pts, src_clean), dim=-1)[0]
    chamfer_dist = torch.mean(dist_src, dim=0) + torch.mean(dist_ref, dim=0)
    return chamfer_dist


def test_crop():
    modelnet_test = Modelnet40Pair(
        path=dataset_path, r_lrf=r_lrf, theta_space=theta_space,
        mode="test", pre_encode_path=pre_encode_path, noise_type="crop"
    )
    net = SAO()
    net.to(device)
    sao_param_path = "./params/sao-modelnet-6space-circle-in.pth"
    net.load_state_dict(torch.load(sao_param_path))
    net.eval()
    chamfer_test, test_cnt = 0, 0
    valid_idx = list(np.load("./test_idx.npy"))
    with torch.no_grad():
        for i in range(len(modelnet_test)):
            source_pts, source_normal, source_pre_encode, target_pts, target_normal, target_pre_encode, match, raw_pts, T, idx = modelnet_test[i]
            if idx not in valid_idx:
                continue
            test_cnt += 1
            source_pts_norm, target_pts_norm = pc_normalize(deepcopy(source_pts)), pc_normalize(deepcopy(target_pts))
            source_inp = torch.Tensor(np.concatenate([source_pts_norm, source_pts_norm, source_pre_encode], axis=1)).to(device)
            target_inp = torch.Tensor(np.concatenate([target_pts_norm, source_pts_norm, target_pre_encode], axis=1)).to(device)
            # network predict
            source_f, target_f, source_overlap, target_overlap = net(source_inp, target_inp)
            source_f, target_f, source_overlap, target_overlap = source_f.cpu().numpy(), target_f.cpu().numpy(), source_overlap.cpu().numpy(), target_overlap.cpu().numpy()
            thresh = 0.4
            source_overlap_idx, target_overlap_idx = np.nonzero(source_overlap >= thresh)[0], np.nonzero(target_overlap >= thresh)[0]

            # # 画出预测的重叠部分，看看准不准
            # source_pc = get_pc(source_pts, source_normal, [1, 0.706, 0])
            # target_pc = get_pc(target_pts, target_normal, [0, 0.651, 0.929])
            # source_colors, target_colors = np.asarray(source_pc.colors), np.asarray(target_pc.colors)
            # source_colors[source_overlap_idx, :], target_colors[target_overlap_idx, :] = np.array([[1, 0, 0]] * source_overlap_idx.shape[0]), np.asarray([[0, 0, 1]] * target_overlap_idx.shape[0])
            # o3d.draw_geometries([deepcopy(source_pc).transform(T), target_pc], window_name="test", width=1000, height=800)

            source_pc = get_pc(source_pts, source_normal, [1, 0.706, 0])
            target_pc = get_pc(target_pts, target_normal, [0, 0.651, 0.929])
            # 重叠部分配准
            source_overlap_pc, target_overlap_pc = o3d.PointCloud(), o3d.PointCloud()
            source_overlap_pc.points, target_overlap_pc.points = o3d.Vector3dVector(np.asarray(source_pc.points)[source_overlap_idx]), o3d.Vector3dVector(np.asarray(target_pc.points)[target_overlap_idx])

            ransac_T = ransac_pose_estimation(
                source_pts[source_overlap_idx], target_pts[target_overlap_idx],
                source_f[source_overlap_idx], target_f[target_overlap_idx], distance_threshold=0.07)
            icp_result = o3d.registration_icp(source_overlap_pc, target_overlap_pc, 0.06, init=ransac_T)
            icp_T = icp_result.transformation

            # 评估
            chamfer_dist = chamfer_distance(source_pts, target_pts, raw_pts, icp_T, T)
            chamfer_test += chamfer_dist.item()

            print("\rtest process: %s  cur chamfer dis: %.5f   item dis: %.5f" % (processbar(test_cnt, len(valid_idx)), chamfer_test / test_cnt, chamfer_dist.item()), end="")
            # o3d.draw_geometries([source_pc, target_pc], window_name="before registration", width=1000, height=800)
            # o3d.draw_geometries([source_pc.transform(icp_T), target_pc], window_name="registration", width=1000, height=800)
    print("\ntest finish, charmer dis: %.5f" % (chamfer_test / len(valid_idx)))


def test_clean():
    modelnet_test = Modelnet40Pair(
        path=dataset_path, r_lrf=r_lrf, theta_space=1 / 9,
        mode="test", pre_encode_path=None, noise_type="clean"
    )
    chamfer_test, test_cnt = 0, 0
    valid_idx = list(np.load("./test_idx.npy"))
    for i in range(len(modelnet_test)):
        if i not in valid_idx:
            continue
        test_cnt += 1
        source_pts, source_normal, source_pre_encode, target_pts, target_normal, target_pre_encode, match, raw_pts, T, idx = modelnet_test[i]
        source_pc = get_pc(source_pts, source_normal, [1, 0.706, 0])
        target_pc = get_pc(target_pts, target_normal, [0, 0.651, 0.929])

        ransac_T = ransac_pose_estimation(
            source_pts, target_pts,
            source_pre_encode, target_pre_encode,
            distance_threshold=0.07, max_iter=50000, max_valid=5000
        )
        icp_result = o3d.registration_icp(source_pc, target_pc, 0.04, init=ransac_T)
        icp_T = icp_result.transformation

        # 评估
        chamfer_dist = chamfer_distance(source_pts, target_pts, raw_pts, icp_T, T)
        chamfer_test += chamfer_dist.item()

        print("\rtest process: %s  cur chamfer dis: %.5f   item dis: %.5f" % (
        processbar(test_cnt, len(valid_idx)), chamfer_test / test_cnt, chamfer_dist.item()), end="")
        # o3d.draw_geometries([source_pc, target_pc], window_name="before registration", width=1000, height=800)
        # o3d.draw_geometries([source_pc.transform(icp_T), target_pc], window_name="registration", width=1000, height=800)
    print("\ntest finish, charmer dis: %.5f" % (chamfer_test / len(valid_idx)))


def test_jitter():
    modelnet_test = Modelnet40Pair(
        path=dataset_path, r_lrf=r_lrf, theta_space=1 / 6,
        mode="test", pre_encode_path=None, noise_type="jitter"
    )
    chamfer_test, test_cnt = 0, 0
    valid_idx = list(np.load("./test_idx.npy"))
    for i in range(len(modelnet_test)):
        if i not in valid_idx:
            continue
        test_cnt += 1
        source_pts, source_normal, source_pre_encode, target_pts, target_normal, target_pre_encode, match, raw_pts, T, idx = modelnet_test[i]
        source_pc = get_pc(source_pts, source_normal, [1, 0.706, 0])
        target_pc = get_pc(target_pts, target_normal, [0, 0.651, 0.929])

        ransac_T = ransac_pose_estimation(
            source_pts, target_pts,
            source_pre_encode, target_pre_encode,
            distance_threshold=0.07, max_iter=50000, max_valid=5000
        )
        icp_result = o3d.registration_icp(source_pc, target_pc, 0.04, init=ransac_T)
        icp_T = icp_result.transformation

        # 评估
        chamfer_dist = chamfer_distance(source_pts, target_pts, raw_pts, icp_T, T)
        chamfer_test += chamfer_dist.item()

        print("\rtest process: %s  cur chamfer dis: %.5f   item dis: %.5f" % (
        processbar(test_cnt, len(valid_idx)), chamfer_test / test_cnt, chamfer_dist.item()), end="")
        # o3d.draw_geometries([source_pc, target_pc], window_name="before registration", width=1000, height=800)
        # o3d.draw_geometries([source_pc.transform(icp_T), target_pc], window_name="registration", width=1000, height=800)
    print("\ntest finish, charmer dis: %.5f" % (chamfer_test / len(valid_idx)))


if __name__ == '__main__':
    # test_crop()
    # test_clean()
    test_jitter()