import numpy as np
import torch
import math


def compute(pc, x_theta_space=1/6, z_theta_space=1/6):
    # theta_space 是多少多少π，1/6就是间隔为π/6的意思
    # x1 y1 z1
    # x2 y2 z2
    # ...
    # xn yn zn
    pc = torch.Tensor(pc)
    x_theta = torch.atan2(pc[:, 1], pc[:, 0])
    # 0 ~ 2π
    x_theta[x_theta < 0] = 2*math.pi + x_theta[x_theta < 0]
    # 0 ~ π
    z_theta = torch.atan2(torch.sqrt(pc[:, 0]**2+pc[:, 1]**2), pc[:, 2])
    # x 的角度总共划分为多少个扇形
    x_theta_kind, z_theta_kind = int(2 / x_theta_space), int(1 / z_theta_space)
    x_index, z_index = torch.floor(x_theta / (x_theta_space*math.pi)), torch.floor(z_theta / (z_theta_space*math.pi))
    x_index, z_index = z_index.long(), z_index.long()

    x_index[x_index > x_theta_kind-1], z_index[z_index > z_theta_kind-1] = x_theta_kind-1, z_theta_kind-1
    x_index[x_index < 0], z_index[z_index < 0] = 0, 0

    index = x_index * z_theta_kind + z_index
    value = torch.sqrt(pc[:, 0]**2+pc[:, 1]**2+pc[:, 2]**2)

    index, value = index.numpy(), value.numpy()

    max_ind = np.lexsort([value, index])
    desc = np.zeros((x_theta_kind*z_theta_kind, ))
    desc[index[max_ind]] = value[max_ind]

    return desc


if __name__ == '__main__':
    pass
