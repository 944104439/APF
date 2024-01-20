import torch

import os


from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from torch import nn
import torch.nn.functional as F

from PromptModels.pointnet_seg import PointNet
from PromptModels.pointnetv2 import PointNetv2
from PromptModels.pointnext import pointNext
from PromptModels.utils import Block


class DGCNN_Propagation(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        # print('using group version 2')
        self.k = k
        from knn_cuda import KNN
        self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(nn.Conv2d(1536, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(1024, 768, kernel_size=1, bias=False),
                                   nn.GroupNorm(
                                       4, 768),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, coor, f, coor_q, f_q):
        """ coor, f : B 3 G ; B C G
            coor_q, f_q : B 3 N; B 3 N
        """
        # dgcnn upsample
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
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


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
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
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))


        return new_points


def fps(data, number):
    '''
        data B N C
        number int
    '''
    fps_idx = furthest_point_sample(data[:, :, :3].contiguous(), number)
    fps_data = torch.gather(
        data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    return fps_data


class PointFormerSeg(nn.Module):

    def __init__(self,num_classes=40, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,act_layer=nn.GELU, npoint=196, radius=0.15, nsample=64,normal=False):
        # Recreate ViT
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.embed = embed_dim
        self.enc_norm = norm_layer(embed_dim)
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.normal = normal

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        self.num_classes = num_classes

        self.encoder = pointNext()
        # self.encoder = PointNetv2(in_channel=14)
        self.trans_dim = 768

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, self.embed)
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(512 + 768, 256, 1, bias=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Conv1d(256, num_classes, 1, bias=True)
        )
        # self.reduce_dim = nn.Linear(768, 384)

        self.propagation_2 = PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
        self.propagation_1= PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
        self.propagation_0 = PointNetFeaturePropagation(in_channel= self.trans_dim + 3 + 16, mlp = [self.trans_dim * 4, self.trans_dim])

        self.reduce_dim = 512
        self.propagation_3 = PointNetFeaturePropagation(in_channel=512 + 256,mlp =[self.reduce_dim * 4, self.reduce_dim])
        self.propagation_4 = PointNetFeaturePropagation(in_channel=512 + 128,
                                                        mlp=[self.reduce_dim * 4, self.reduce_dim])
        self.propagation_5 = PointNetFeaturePropagation(in_channel=512 + 64,
                                                        mlp=[self.reduce_dim * 4, self.reduce_dim])
        self.propagation_6 = PointNetFeaturePropagation(in_channel=512 + 32,
                                                        mlp=[self.reduce_dim * 4, self.reduce_dim])
        self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

        for name, param in self.named_parameters():
            if ('encoder' in name or 'adaptmlp' in name or 'pos'
                    in name or 'classifier' in name or 'propagation' in name or
                    'dgcnn' in name or 'enc_norm' in name):
                param.requires_grad = True


    def forward_features(self, x, cls_label):

        B, N, C = x.shape
        pts = x[:, :, :3]
        # pts = pts.transpose(-1, -2)  # B N 3
        # # divide the point clo  ud in the same form. This is important
        # l1_xyz,l1_points,center,group_input_tokens = self.encoder(x)  # B G N
        # pv2_feature = self.propagation_3(l1_xyz.transpose(1,2), center.transpose(1,2), l1_points.transpose(1,2), group_input_tokens.transpose(1,2))
        # pv2_feature2 = self.propagation_4(pts.permute(0,2,1),l1_xyz.permute(0,2,1), x.permute(0,2,1), pv2_feature)
        """
               Input:
                   xyz1: input points position data, [B, C, N]
                   xyz2: sampled input points position data, [B, C, S]
                   points1: input points data, [B, D, N]
                   points2: input points data, [B, D, S]
               Return:
                   new_points: upsampled points data, [B, D', N]
        """
        group_input_tokens,feats = self.encoder(x)
        l0_xyz, l0_points = feats[0][1],feats[0][0]
        l1_xyz, l1_points = feats[1][1], feats[1][0]
        l2_xyz, l2_points = feats[2][1], feats[2][0]
        l3_xyz, l3_points = feats[3][1], feats[3][0]
        l4_xyz, l4_points = feats[4][1], feats[4][0]

        # 196->256
        l3_feature = self.propagation_3(l3_xyz,l4_xyz,l3_points,l4_points)
        # 256->512
        l2_feature = self.propagation_4(l2_xyz,l3_xyz,l2_points,l3_feature)
        l1_feature = self.propagation_5(l1_xyz, l2_xyz, l1_points, l2_feature)
        l0_feature = self.propagation_6(l0_xyz, l1_xyz, l0_points, l1_feature)



        x = group_input_tokens
        center = feats[-1][1].transpose(1,2)
        # trans_group_input_tokens = self.reduce_dim(group_input_tokens) # B G 384
        # add pos embedding
        pos = self.pos_embed(center)
        # # final input
        # transformer
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i in range(len(self.blocks)):
            x = self.blocks[i](x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        # B,C,N
        feature_list = [self.enc_norm(x).transpose(-1, -2).contiguous() for x in feature_list]

        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        center_level_0 = pts.transpose(-1, -2).contiguous()
        f_level_0 = torch.cat([cls_label_one_hot, center_level_0], 1)
        # print(f_level_0.shape)

        center_level_1 = fps(pts, 512).transpose(-1, -2).contiguous()
        f_level_1 = center_level_1
        center_level_2 = fps(pts, 256).transpose(-1, -2).contiguous()
        f_level_2 = center_level_2
        center_level_3 = center.transpose(-1, -2).contiguous()

        # init the feature by 3nn propagation
        f_level_3 = feature_list[2]
        f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list[1])
        f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, feature_list[0])
        # print(center_level_3.shape,f_level_3.shape,center_level_2.shape,f_level_2.shape)
        # bottom up
        f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
        f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
        f_level_0 = self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1)
        # print(f_level_0.shape)
        f_level_0 = torch.cat([f_level_0,l0_feature],dim = 1)
        x = self.classifier(f_level_0)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x, cls_label):
        x = self.forward_features(x, cls_label)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss

def pointFormerSeg(num_classes=50, base_model='vit_base_patch16_224_in21k'):

    import timm
    basic_model = timm.create_model(base_model, pretrained=True)
    base_state_dict = basic_model.state_dict()
    del base_state_dict['head.weight']
    del base_state_dict['head.bias']
    model = PointFormerSeg(num_classes=num_classes,depth=12, num_heads=12, embed_dim=768)
    model.load_state_dict(base_state_dict, False)
    model.freeze()
    return model

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()

    return new_y

if __name__ == '__main__':
    pass
    #
    # num_classes = 16
    # num_part = 50
    # B = 2
    # points = torch.randn(2, 2048, 7).to('cuda')
    # label = torch.randint(0, num_classes, (B, 1)).to('cuda')
    # # print(label.shape)
    # model = build_promptmodel_seg(num_classes=num_part).to('cuda')
    # for name, param in model.named_parameters():
    #     print(name)
    # print("======= hot ========")
    # for name, param in model.named_parameters():
    #     if param.requires_grad is True:
    #         print(f"{name} : {param.shape}")
    # seg_pred = model(points,to_categorical(label,num_classes))
    # print(to_categorical(label,num_classes).shape)
    # print(seg_pred.shape)
