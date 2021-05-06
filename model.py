import torch
from torch import nn
from torch.nn import functional as F
import math
from utils import square_distance, PointNetSetAbstraction, UpSample, PointNetSetAbstractionMsg
from loss import get_weighted_bce_loss, get_circle_loss

cpu, gpu = "cpu", "cuda:0"
device = torch.device(gpu)


class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()

    def forward(self, x, keepdim=True):
        return torch.max(x, dim=2, keepdim=keepdim)[0]


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, relu=True, bn=True):
        super(MLP, self).__init__()
        mlp = [nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)]
        if bn:
            mlp.append(nn.BatchNorm2d(out_channel))
        if relu:
            mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.share_mlp = nn.Sequential(MLP(3, 64), MLP(64, 128))
        self.independ_mlp = nn.Sequential(*[nn.Sequential(MLP(128, 1024, relu=False), MaxPool()) for _ in range(16)])
        self.capsule = Capsule(in_cap_size=1024, in_vec_len=16, out_cap_size=32, out_vec_len=32)

    def forward(self, x):
        # x_size:  batch_size x pt_num x dim
        x = self.share_mlp(x.permute([0, 2, 1]).unsqueeze(3))  # batch_size x 128 x pt_num x 1
        x = torch.cat([mlp(x) for mlp in self.independ_mlp], dim=2).squeeze(3).permute([0, 2, 1])  # batch_size x 16 x 1024
        # 对x挤压，把x变成capsule
        x = self.squash(x.permute([0, 2, 1]))
        x = self.capsule(x)  # batch_size x 64 x 64
        return x

    def squash(self, s):
        # batch x cap_size x cap_vel_len
        s_mo = torch.norm(s, p=2, dim=2, keepdim=True)  # batch x cap_size x 1
        return s*(s_mo**2/((1+s_mo**2)*s_mo))


class Capsule(nn.Module):
    def __init__(self, in_cap_size, in_vec_len, out_cap_size, out_vec_len, T=3):
        super(Capsule, self).__init__()
        self.W = nn.Parameter(0.01*torch.randn(out_cap_size, in_cap_size, in_vec_len, out_vec_len))
        self.T = T  # Dynamic Routing 的迭代次数

    def forward(self, x):
        # x_size:  batch_size x in_cap_size x in_cap_len
        u = torch.matmul(x[:, None, :, None, :], self.W).squeeze(3) # batch_size x out_cap_size x in_cap_size x out_vec_len
        u_detach = u.detach()
        b = torch.zeros(u.shape[0], u.shape[1], u.shape[2], 1).to(device)   # batch_size x out_cap_size x in_cap_size x 1
        s = None  # 胶囊里面的s，就是u的加权
        for i in range(self.T):
            c = torch.softmax(b, dim=1)
            if i == self.T-1:
                s = (u*c).sum(dim=2)
            else:
                a = self.squash((u*c).sum(dim=2))
                b = b + (a.unsqueeze(2)*u_detach).sum(3).unsqueeze(3)
        v = self.squash(s)
        return v

    def squash(self, s):
        # batch x cap_size x cap_vel_len
        s_mo = torch.norm(s, p=2, dim=2, keepdim=True)  # batch x cap_size x 1
        return s*(s_mo**2/((1+s_mo**2)*s_mo))


class Decoder(nn.Module):
    def __init__(self, point_num=1024, cap_size=32):
        super(Decoder, self).__init__()
        self.decoder_mlp_num = point_num // cap_size
        self.mlp = nn.Sequential(*[nn.Sequential(
            MLP(34, 64), MLP(64, 32), MLP(32, 16), MLP(16, 3, bn=False, relu=False), nn.Tanh()
        ) for _ in range(self.decoder_mlp_num)])

    def forward(self, x):
        # x_size :  batch_size x cap_num x cap_vec_len
        rand_grid = torch.FloatTensor(x.shape[0], 2, x.shape[1]).to(device)
        rand_grid.data.uniform_(0, 1)
        outs = []
        for mlp in self.mlp:
            #               batch_size x 3 x 64 x 1                                           batch_size x 3 x 64
            outs.append(mlp(torch.cat([x.permute([0, 2, 1]), rand_grid], dim=1).unsqueeze(3)).squeeze(3))
        reconstruct_points = torch.stack(outs, dim=2).permute([0, 1, 3, 2])
        return reconstruct_points.contiguous().view(x.shape[0], 3, -1).permute([0, 2, 1])


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(point_num=1024)

    def forward(self, x):
        # x_size: batch_size x point_num x 3
        codeword = self.encoder(x)
        reconstruct_points = self.decoder(codeword)
        return codeword.view(x.shape[0], -1), reconstruct_points


class MutiHeadAttention(nn.Module):
    def __init__(self, n_head, in_channel, qk_channel, v_channel, out_channel, mid_channel, feedforward=True):
        super(MutiHeadAttention, self).__init__()
        # mutihead attention
        self.n_head = n_head
        self.qk_channel, self.v_channel = qk_channel, v_channel
        self.WQ = nn.Linear(in_channel, qk_channel*n_head, bias=False)
        self.WK = nn.Linear(in_channel, qk_channel*n_head, bias=False)
        self.WV = nn.Linear(in_channel, v_channel*n_head, bias=False)
        self.linear = nn.Linear(v_channel*n_head, out_channel, bias=False)
        # 不确定要不要仿射变换，先不加试试
        self.norm1 = nn.LayerNorm(out_channel, elementwise_affine=False)
        # feedforward
        self.feedforward = feedforward
        if self.feedforward:
            self.feedforward_layer = nn.Sequential(
                nn.Linear(out_channel, mid_channel, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mid_channel, out_channel, bias=False)
            )
            self.norm2 = nn.LayerNorm(out_channel, elementwise_affine=False)

    def forward(self, query, key, value, mask=None):
        # q, k, v: batch_size x n x in_channel
        # mask： batch_size x n x n
        batch_size = query.shape[0]
        # batch_size x n x in_channel  -->  batch_size x n x v_channel
        Q = self.WQ(query).view(batch_size, -1, self.n_head, self.qk_channel).transpose(1, 2)  # batch_size x n_head x n x q_channel
        K = self.WK(key).view(batch_size, -1, self.n_head, self.qk_channel).transpose(1, 2)    # batch_size x n_head x n x k_channel
        V = self.WV(value).view(batch_size, -1, self.n_head, self.v_channel).transpose(1, 2)   # batch_size x n_head x n x v_channel
        weight = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.qk_channel)               # batch_size x n_head x n x n
        if mask is None:
            weight = torch.softmax(weight, dim=3)
        else:
            weight = torch.softmax(weight + mask.unsqueeze(1), dim=3)                              # batch_size x n_head x n x v_channel
        # print(weight.dtype, V.dtype, mask.dtype)
        out = torch.matmul(weight, V).transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.v_channel)
        out = self.linear(out)                                                                 # batch_size x n x out_channel
        out = self.norm1(query + out)
        if self.feedforward:
            return self.norm2(out + self.feedforward_layer(out))
        else:
            return out


class SelfAggregation(nn.Module):
    def __init__(self):
        super(SelfAggregation, self).__init__()
        # downsample 1
        DownSample = PointNetSetAbstraction
        self.ds1 = DownSample(npoint=358, radius=0.2, nsample=64, in_channel=162+3, mlp=[1024, 512, 512], group_all=False, cat_xyz=True)
        # downsample 2
        self.ds2 = DownSample(npoint=128, radius=0.4, nsample=64, in_channel=512+3, mlp=[512, 512, 256], group_all=False, cat_xyz=True)
        # downsample 3
        self.ds3 = DownSample(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True, cat_xyz=True)

        self.up3 = UpSample(in_channel=1280, mlp=[512, 512])
        self.up2 = UpSample(in_channel=512+512, mlp=[512, 256])
        self.up1 = UpSample(in_channel=256+162, mlp=[512, 256, 128])

    def forward(self, x):
        # x: batch_size x n x d
        # downsample
        x = x.permute([0, 2, 1])
        l0_points, l0_xyz = x[:, 3:, :], x[:, :3, :]
        l1_xyz, l1_points = self.ds1(l0_xyz, l0_points)                         # batch_size x n x 3, batch_size x n x d
        l2_xyz, l2_points = self.ds2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.ds3(l2_xyz, l2_points)
        # upsample
        l2_points = self.up3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.up2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.up1(l0_xyz, l1_xyz, l0_points, l1_points)
        return l0_points.permute([0, 2, 1])


class SelfAggregationMsg(nn.Module):
    def __init__(self):
        super(SelfAggregationMsg, self).__init__()
        DownSample = PointNetSetAbstractionMsg
        self.ds1 = DownSample(358, [0.1, 0.2, 0.4], [32, 64, 128], 72, [[32, 32, 64], [64, 64, 128], [64, 96, 128]], bn=False)
        # downsample 2
        self.ds2 = DownSample(128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]], bn=False)
        # downsample 3
        self.ds3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512+3, mlp=[256, 512, 1024], group_all=True, bn=False)

        # self.overlap_atte = MutiHeadAttention(n_head=4, in_channel=256, qk_channel=64, v_channel=64, out_channel=256,mid_channel=1024)

        self.up3 = UpSample(in_channel=1024+256+256, mlp=[256, 256], bn=False)
        self.up2 = UpSample(in_channel=256+128+128+64, mlp=[256, 128], bn=False)
        self.up1 = UpSample(in_channel=128 + 72 + 0, mlp=[128, 256], bn=False)

    def forward(self, x):
        x = x.permute([0, 2, 1])
        l0_points, l0_xyz = x[:, 3:, :], x[:, :3, :]
        l1_xyz, l1_points = self.ds1(l0_xyz, l0_points)  # batch_size x n x 3, batch_size x n x d
        l2_xyz, l2_points = self.ds2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.ds3(l2_xyz, l2_points)
        # upsample
        l2_points = self.up3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.up2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.up1(l0_xyz, l1_xyz, l0_points, l1_points)
        return l0_points.permute([0, 2, 1])


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        # downsample 1
        DownSample = PointNetSetAbstraction
        self.ds1 = DownSample(npoint=358, radius=0.2, nsample=64, in_channel=6+0, mlp=[64, 64, 128], group_all=False)
        # downsample 2
        self.ds2 = DownSample(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        # downsample 3
        self.ds3 = DownSample(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.overlap_atte = MutiHeadAttention(n_head=4, in_channel=256, qk_channel=64, v_channel=64, out_channel=256, mid_channel=1024)

        self.up3 = UpSample(in_channel=1280, mlp=[256, 256])
        self.up2 = UpSample(in_channel=384, mlp=[256, 128])
        self.up1 = UpSample(in_channel=128+6+0, mlp=[512, 256, 256])

    def forward(self, x):
        # x: batch_size x n x d
        # downsample
        x = x.permute([0, 2, 1])
        l0_points, l0_xyz = x, x[:, :3, :]
        l1_xyz, l1_points = self.ds1(l0_xyz, l0_points)  # batch_size x n x 3, batch_size x n x d
        l2_xyz, l2_points = self.ds2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.ds3(l2_xyz, l2_points)
        # upsample
        l2_points = self.up3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.shape)
        l2_points = l2_points.permute([0, 2, 1])
        l2_points = self.overlap_atte(l2_points, l2_points, l2_points, None)
        l2_points = l2_points.permute([0, 2, 1])
        # print(l2_points.shape)
        l1_points = self.up2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.up1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], dim=1), l1_points)
        return l0_points.permute([0, 2, 1])


class PointNetMsg(nn.Module):
    def __init__(self):
        super(PointNetMsg, self).__init__()
        DownSample = PointNetSetAbstractionMsg
        self.ds1 = DownSample(358, [0.1, 0.2, 0.4], [32, 64, 128], 3+0, [[32, 32, 64], [64, 64, 128], [64, 96, 128]], bn=False)
        # downsample 2
        self.ds2 = DownSample(128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]], bn=False)
        # downsample 3
        self.ds3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True, bn=False)

        self.overlap_atte = MutiHeadAttention(n_head=4, in_channel=256, qk_channel=64, v_channel=64, out_channel=256,
                                              mid_channel=1024)

        self.up3 = UpSample(in_channel=1024+256+256, mlp=[256, 256], bn=False)
        self.up2 = UpSample(in_channel=256+128+128+64, mlp=[256, 128], bn=False)
        self.up1 = UpSample(in_channel=128 + 6 + 0, mlp=[128, 256], bn=False)

    def forward(self, x):
        # x: batch_size x n x d
        # downsample
        x = x.permute([0, 2, 1])
        l0_points, l0_xyz = x, x[:, :3, :]
        l1_xyz, l1_points = self.ds1(l0_xyz, l0_points)  # batch_size x n x 3, batch_size x n x d
        l2_xyz, l2_points = self.ds2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.ds3(l2_xyz, l2_points)
        # upsample
        l2_points = self.up3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.shape)
        l2_points = l2_points.permute([0, 2, 1])
        l2_points = self.overlap_atte(l2_points, l2_points, l2_points, None)
        l2_points = l2_points.permute([0, 2, 1])
        # print(l2_points.shape)
        l1_points = self.up2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.up1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], dim=1), l1_points)
        return l0_points.permute([0, 2, 1])


class SAO(nn.Module):
    def __init__(self):
        super(SAO, self).__init__()
        # self.inp_emb = SelfAggregation()
        self.inp_emb = SelfAggregationMsg()
        # self.point_net = PointNet()
        self.point_net = PointNetMsg()
        self.atte = nn.ModuleList([
            MutiHeadAttention(n_head=4, in_channel=256, qk_channel=64, v_channel=64, out_channel=256, mid_channel=1024)
            for _ in range(1)
        ])

        self.fc = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        # self.f_cat_linear = nn.Sequential(
        #     nn.Linear(1024+256, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 256)
        # )

    def forward(self, source_inp, target_inp):
        # f = self.inp_emb(torch.cat([source_inp, target_inp], dim=0))
        # print(f.shape)
        ptnet_source_inp, source_inp = source_inp[:, :3], source_inp[:, 3:]
        ptnet_target_inp, target_inp = target_inp[:, :3], target_inp[:, 3:]
        # print(ptnet_source_inp.shape, source_inp.shape)
        f_out = self.inp_emb(torch.stack([source_inp, target_inp], dim=0))
        # source_f, target_f = self.inp_emb(source_inp.unsqueeze(0)), self.inp_emb(target_inp.unsqueeze(0))
        source_f, target_f = f_out[0], f_out[1]
        overlap_out = self.point_net(torch.stack([ptnet_source_inp, ptnet_target_inp], dim=0))
        # overlap_source_f, overlap_target_f = self.point_net(ptnet_source_inp.unsqueeze(0)), self.point_net(ptnet_target_inp.unsqueeze(0))
        overlap_source_f, overlap_target_f = overlap_out[0].unsqueeze(0), overlap_out[1].unsqueeze(0)
        tf = overlap_target_f
        for i in range(len(self.atte)):
            tf = self.atte[i](overlap_source_f, tf, tf, None)
        sf = overlap_source_f
        for i in range(len(self.atte)):
            sf = self.atte[i](overlap_target_f, sf, sf, None)
        # source_overlap = self.atte(source_f, target_f, target_f, None)[0]
        # target_overlap = self.atte(target_f, source_f, source_f, None)[0]
        source_overlap, target_overlap = tf[0], sf[0]
        source_overlap, target_overlap = self.fc(source_overlap).view(-1), self.fc(target_overlap).view(-1)
        # print(source_inp.shape, source_f[0].shape)
        # source_f = self.f_cat_linear(torch.cat([source_inp[:, 3:], source_f[0]], dim=1))
        # target_f = self.f_cat_linear(torch.cat([target_inp[:, 3:], target_f[0]], dim=1))
        # print(source_overlap.shape, target_overlap.shape)
        # return source_f[0], target_f[0], source_overlap, target_overlap
        # source_f, target_f = F.normalize(source_f, p=2, dim=1), F.normalize(target_f, p=2, dim=1)
        return source_f, target_f, source_overlap, target_overlap


class SAOLoss(nn.Module):
    def __init__(self):
        super(SAOLoss, self).__init__()
        self.bce = get_weighted_bce_loss
        self.circle = get_circle_loss

    def forward(self, model, source_inp, target_inp, match_label, coords_dist):
        # source_inp: n x 1024
        # target_inp: m x 1024
        source_f, target_f, source_overlap, target_overlap = model(source_inp, target_inp)
        source_pos_idx = (match_label.sum(dim=1) > 0)
        target_pos_idx = (match_label.permute([1, 0]).sum(dim=1) > 0)

        # print()
        # print(source_pos_num, target_pos_num)
        # source的重叠区域标签
        source_overlap_label = torch.zeros((source_f.shape[0], )).to(source_f.device)
        source_overlap_label[source_pos_idx] = 1
        # target的重叠区域标签
        target_overlap_label = torch.zeros((target_f.shape[0], )).to(target_f.device)
        target_overlap_label[target_pos_idx] = 1
        # print()
        # print(source_overlap_label.sum(), source_overlap_label.shape[0], target_overlap_label.sum(), target_overlap_label.shape[0])
        source_overlap_loss, source_acc, _ = self.bce(source_overlap, source_overlap_label)
        target_overlap_loss, target_acc, _ = self.bce(target_overlap, target_overlap_label)

        # overlap_match_label = match_label[source_overlap_fps_idx, :]
        # overlap_match_label = overlap_match_label[:, target_overlap_fps_idx]
        feat_dist = torch.sqrt(square_distance(source_f.unsqueeze(0), target_f.unsqueeze(0), normalised=False)[0])
        # match_loss, _, _, _ = hardest_contrastive(source_f[source_overlap_fps_idx], target_f[target_overlap_fps_idx], overlap_match_label)
        match_loss = self.circle(coords_dist, feat_dist, pos_radius=0.04, safe_radius=0.06, log_scale=64)
        if torch.isnan(match_loss):
            print("match loss is nan")
        loss = match_loss + source_overlap_loss + target_overlap_loss

        overlap_thresh = 0.5
        source_pred_pos_idx = (source_overlap >= overlap_thresh)
        target_pred_pos_idx = (target_overlap >= overlap_thresh)

        # 统计正确配对数
        source_pos_f = source_f[source_pred_pos_idx]
        target_pos_f = target_f[target_pred_pos_idx]
        label = match_label[source_pred_pos_idx, :]
        label = label[:, target_pred_pos_idx]
        if source_pos_f.shape[0] == 0 or target_pos_f.shape[0] == 0:
            return loss, source_acc, target_acc, 0
        f_dis_min_idx = square_distance(source_pos_f.unsqueeze(0), target_pos_f.unsqueeze(0))[0].topk(k=1, largest=False, dim=1)[1].view(-1)
        correct = 0
        for i in range(label.shape[0]):
            if label[i, f_dis_min_idx[i]] == 1:
                correct += 1
        match_acc = correct / source_pos_f.shape[0] if source_pos_f.shape[0] != 0 else 0

        return loss, source_acc, target_acc, match_acc


if __name__ == '__main__':
    device = torch.device("cuda:0")
    # ae = AutoEncoder().to(device)
    # x = torch.rand(2, 1024, 3).to(device)
    # codeword, y = ae(x)
    # print(codeword.shape, y.shape)

    # net = SelfAggregation()
    # net.to(device)
    # x = torch.rand(4096, 1027).unsqueeze(0).to(device)
    # y = net(x)
    # print(y.shape)

    net = PointNet()
    net.to(device)
    x = torch.rand(4096, 6).unsqueeze(0).to(device)
    y = net(x)
    print(y.shape)

    # net = SAO()
    # net.to(device)
    # x1 = torch.rand(4096, 1027+3).to(device)
    # x2 = torch.rand(2600, 1027+3).to(device)
    # y1, y2, y3, y4 = net(x1, x2)
    # print(y1.shape, y2.shape, y3.shape, y4.shape)