import numpy as np
import open3d as o3d
import torch
from copy import deepcopy
from dataset import Modelnet40Pair
import utils
from utils import pc_normalize
from model import SAOLoss, SAO, device


r_lrf = 0.4
theta_space = 1 / 6

pre_encode_path = "./modelnet_pre_encode/test"
modelnet_train = Modelnet40Pair(path="E:/modelnet40_ply_hdf5_2048", r_lrf=r_lrf, theta_space=theta_space, mode="train", pre_encode_path=None)
modelnet_test = Modelnet40Pair(path="E:/modelnet40_ply_hdf5_2048", r_lrf=r_lrf, theta_space=theta_space, mode="test", pre_encode_path=pre_encode_path)


def processbar(current, totle):
    process_str = ""
    for i in range(int(20*current/totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|  %d / %d" % (process_str, current, totle)


def train():
    epoch = 251
    lr = 0.001
    min_lr = 0.00001
    lr_update_step = 20
    loss_fn = SAOLoss()
    net = SAO()
    net.to(device)
    sao_param_path = "./params/sao-modelnet-6space-circle-in.pth"
    net.load_state_dict(torch.load(sao_param_path))
    # optimizer = torch.optim.SGD(params=net.parameters(), lr=lr, weight_decay=0)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=0)

    def update_lr(optimizer, gamma=0.5):
        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        lr = max(lr * gamma, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("lr update finished  cur lr: %.5f" % lr)

    def evaluate():
        mean_source_acc, mean_target_acc, mean_match_acc, process = 0, 0, 0, 0
        with torch.no_grad():
            for i in range(len(modelnet_test)):
                source_pts, source_normal, source_pre_encode, target_pts, target_normal, target_pre_encode, match, raw_pts, T, idx = modelnet_test[i]
                source_pts_norm, target_pts_norm = pc_normalize(deepcopy(source_pts)), pc_normalize(deepcopy(target_pts))
                source_inp = torch.Tensor(np.concatenate([source_pts_norm, source_pts_norm, source_pre_encode], axis=1)).to(device)
                target_inp = torch.Tensor(np.concatenate([target_pts_norm, target_pts_norm, target_pre_encode], axis=1)).to(device)
                match_label = torch.Tensor(match).to(device)
                # 为circle loss做准备
                source_pc = o3d.PointCloud()
                source_pc.points = o3d.Vector3dVector(source_pts)
                coords_dist = utils.square_distance(torch.Tensor(np.asarray(deepcopy(source_pc).transform(T).points)).unsqueeze(0), torch.Tensor(target_pts).unsqueeze(0))[0].to(device)
                coords_dist = torch.sqrt(coords_dist)
                # print(((coords_dist < 0.04).sum(1) > 0).sum(), (match_label.sum(1) > 0).sum())
                loss, source_acc, target_acc, match_acc = loss_fn(net, source_inp, target_inp, match_label, coords_dist)

                process += 1
                mean_source_acc += source_acc
                mean_target_acc += target_acc
                mean_match_acc += match_acc
                print("\rtest process: %s   loss: %.5f   source overlap acc: %.5f  target overlap acc: %.5f  match acc: %.5f" % (processbar(process, len(modelnet_test)), loss.item(), source_acc, target_acc, match_acc), end="")
        mean_source_acc /= len(modelnet_test)
        mean_target_acc /= len(modelnet_test)
        mean_match_acc /= len(modelnet_test)
        print("\ntest finish  mean source overlap acc: %.5f  mean target overlap acc: %.5f  mean match acc: %.5f" % (mean_source_acc, mean_target_acc, mean_match_acc))
        return mean_match_acc

    max_acc = 0
    for epoch_count in range(1, epoch + 1):
        mean_source_acc, mean_target_acc, mean_match_acc, process = 0, 0, 0, 0
        loss_val = 0
        rand_idx = np.random.permutation(len(modelnet_train))
        for i in range(len(modelnet_train)):
            source_pts, source_normal, source_pre_encode, target_pts, target_normal, target_pre_encode, match, raw_pts, T, idx = modelnet_train[rand_idx[i]]
            source_pts_norm, target_pts_norm = pc_normalize(deepcopy(source_pts)), pc_normalize(deepcopy(target_pts))
            source_inp = torch.Tensor(np.concatenate([source_pts_norm, source_pts_norm, source_pre_encode], axis=1)).to(device)
            target_inp = torch.Tensor(np.concatenate([target_pts_norm, target_pts_norm, target_pre_encode], axis=1)).to(device)
            match_label = torch.Tensor(match).to(device)
            # 为circle loss做准备
            source_pc = o3d.PointCloud()
            source_pc.points = o3d.Vector3dVector(source_pts)
            coords_dist = utils.square_distance(torch.Tensor(np.asarray(deepcopy(source_pc).transform(T).points)).unsqueeze(0), torch.Tensor(target_pts).unsqueeze(0))[0].to(device)
            coords_dist = torch.sqrt(coords_dist)
            # print(((coords_dist < 0.04).sum(1) > 0).sum(), (match_label.sum(1) > 0).sum())
            loss, source_acc, target_acc, match_acc = loss_fn(net, source_inp, target_inp, match_label, coords_dist)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val += loss.item()
            process += 1
            mean_source_acc += source_acc
            mean_target_acc += target_acc
            mean_match_acc += match_acc
            print("\rprocess: %s   loss: %.5f   source overlap acc: %.5f  target overlap acc: %.5f  match acc: %.5f" % (processbar(process, len(modelnet_train)), loss.item(), source_acc, target_acc, match_acc), end="")
        mean_source_acc /= len(modelnet_train)
        mean_target_acc /= len(modelnet_train)
        mean_match_acc /= len(modelnet_train)
        print("\nepoch: %d  loss: %.5f  mean source overlap acc: %.5f  mean target overlap acc: %.5f  mean match acc: %.5f" % (epoch_count, loss_val, mean_source_acc, mean_target_acc, mean_match_acc))
        test_match_acc = evaluate()
        if max_acc < test_match_acc:
            max_acc = test_match_acc
            print("save ....")
            torch.save(net.state_dict(), sao_param_path)
            print("finish !!!!!!")
        if epoch_count % lr_update_step == 0:
            update_lr(optimizer, 0.5)


if __name__ == '__main__':
    train()