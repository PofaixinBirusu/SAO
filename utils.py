import numpy as np
import open3d as o3d
import torch
from torch import nn
from torch.nn import functional as F


def fps(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def voxel_downsample(pts, voxel_size):
    pc = o3d.PointCloud()
    # print(pts.shape)
    pc.points = o3d.Vector3dVector(pts.cpu().numpy())
    v_pc = o3d.voxel_down_sample(pc, voxel_size=voxel_size)
    v_pts = torch.Tensor(np.asarray(v_pc.points)).to(pts.device)
    idx = square_distance(v_pts.unsqueeze(0), pts.unsqueeze(0))[0].topk(k=1, dim=1, largest=False)[1].view(-1)
    return idx


def pc_normalize(pc, scale_=False):
    pmin = np.min(pc, axis=0)
    pmax = np.max(pc, axis=0)
    pc -= (pmin + pmax) / 2
    if scale_:
        scale = np.max(np.linalg.norm(pc, axis=1))
        pc *= 1.0 / scale
    return pc


# def square_distance(src, dst):
#     B, N, _ = src.shape
#     _, M, _ = dst.shape
#     dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
#     dist += torch.sum(src ** 2, -1).view(B, N, 1)
#     dist += torch.sum(dst ** 2, -1).view(B, 1, M)
#     return dist
def square_distance(src, dst, normalised=False):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if normalised:
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
        dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, voxel_size=None, cat_xyz=True):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    if voxel_size is None:
        fps_idx = fps(xyz, npoint)
    else:
        fps_idx = voxel_downsample(xyz[0], voxel_size).unsqueeze(0)
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, fps_idx[0].shape[0], 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        if cat_xyz:
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points, cat_xyz=True):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        if cat_xyz:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = points.view(B, 1, N, -1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

# ================================= 配准相关 =================================
def to_array(tensor):
    if not isinstance(tensor, np.ndarray):
        if tensor.device == torch.device('cpu'):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_tensor(array):
    if not isinstance(array,torch.Tensor):
        return torch.from_numpy(array).float()
    else:
        return array


def to_o3d_pcd(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd


def to_o3d_feats(embedding):
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats


def ransac_pose_estimation(src_pcd, tgt_pcd, src_feat, tgt_feat, distance_threshold=0.04, ransac_n=3, max_iter=1e6, max_valid=1e5):

    src_pcd = to_o3d_pcd(src_pcd)
    tgt_pcd = to_o3d_pcd(tgt_pcd)
    src_feats = to_o3d_feats(src_feat)
    tgt_feats = to_o3d_feats(tgt_feat)

    result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
        src_pcd, tgt_pcd, src_feats, tgt_feats, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), ransac_n,
        [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.registration.RANSACConvergenceCriteria(max_iter, max_valid))

    return result_ransac.transformation

# =================================== 特征提取基础层 ==============================
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, cat_xyz=True, bn=True):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_norms.append(nn.BatchNorm2d(out_channel) if self.bn else nn.InstanceNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.cat_xyz = cat_xyz

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points, cat_xyz=self.cat_xyz)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, cat_xyz=self.cat_xyz)
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            norm = self.mlp_norms[i]
            new_points = conv(new_points)
            new_points = norm(new_points)
            new_points = F.relu(new_points)

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, bn=True):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn = bn
        self.norm_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            norms = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                # norms.append(nn.BatchNorm2d(out_channel) if self.bn else nn.LayerNorm(out_channel))
                norms.append(nn.BatchNorm2d(out_channel) if self.bn else nn.InstanceNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.norm_blocks.append(norms)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, fps(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                norm = self.norm_blocks[i][j]
                grouped_points = conv(grouped_points)
                grouped_points = norm(grouped_points)
                # if self.bn:
                #     grouped_points = norm(grouped_points)
                # else:
                #     grouped_points = norm(grouped_points.permute([0, 2, 3, 1]).contiguous()).permute([0, 3, 1, 2]).contiguous()
                grouped_points = F.relu(grouped_points)
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class UpSample(nn.Module):
    def __init__(self, in_channel, mlp, bn=True):
        super(UpSample, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            # self.mlp_norms.append(nn.BatchNorm1d(out_channel) if bn else nn.LayerNorm(out_channel))
            self.mlp_norms.append(nn.BatchNorm1d(out_channel) if bn else nn.InstanceNorm1d(out_channel))
            last_channel = out_channel
        self.bn = bn

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            norm = self.mlp_norms[i]
            new_points = conv(new_points)
            new_points = norm(new_points)
            # if self.bn:
            #     new_points = norm(new_points)
            # else:
            #     new_points = norm(new_points.permute([0, 2, 1]).contiguous()).permute([0, 2, 1]).contiguous()
            new_points = F.relu(new_points)
        return new_points
