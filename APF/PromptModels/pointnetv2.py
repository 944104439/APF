import torch
# from knn_cuda import KNN
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from torch import nn



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


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


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


class Encoder_large(nn.Module):  # Embedding module
    def __init__(self, in_channels,encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.in_channels = in_channels
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(self.encoder_channel*2, self.encoder_channel*2, 1),
            nn.BatchNorm1d(self.encoder_channel*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.encoder_channel*2, self.encoder_channel, 1)
        )

    def forward_feature(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, -1)
        # encoder

        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        # print(feature.shape)
        # feature_global = feature.max(2,keepdim=True)[0] + feature.mean(2,keepdim=True)
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        # feature_global = feature.max(2)[0] + feature.mean(2)

        return feature_global.reshape(bs, g, self.encoder_channel)

    def forward(self, point_groups):
        feature = self.forward_feature(point_groups)

        return feature


class Encoder_small(nn.Module):  # Embedding module
    def __init__(self,in_channels, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.in_channels = in_channels
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.in_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, -1)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        from knn_cuda import KNN
        self.knn = KNN(group_size,True)

    def simplied_morton_sorting(self, xyz, center):
        '''
        Simplifying the Morton code sorting to iterate and set the nearest patch to the last patch as the next patch, we found this to be more efficient.

        return : B*group torch.Size([392])
        '''
        batch_size, num_points, _ = xyz.shape
        distances_batch = torch.cdist(center, center)
        distances_batch[:, torch.eye(self.num_group).bool()] = float("inf")
        idx_base = torch.arange(
            0, batch_size, device=xyz.device) * self.num_group
        sorted_indices_list = []
        sorted_indices_list.append(idx_base)
        distances_batch = distances_batch.view(batch_size, self.num_group, self.num_group).transpose(
            1, 2).contiguous().view(batch_size * self.num_group, self.num_group)
        distances_batch[idx_base] = float("inf")
        distances_batch = distances_batch.view(
            batch_size, self.num_group, self.num_group).transpose(1, 2).contiguous()
        for i in range(self.num_group - 1):
            distances_batch = distances_batch.view(
                batch_size * self.num_group, self.num_group)
            distances_to_last_batch = distances_batch[sorted_indices_list[-1]]
            closest_point_idx = torch.argmin(distances_to_last_batch, dim=-1)
            closest_point_idx = closest_point_idx + idx_base
            sorted_indices_list.append(closest_point_idx)
            distances_batch = distances_batch.view(batch_size, self.num_group, self.num_group).transpose(
                1, 2).contiguous().view(batch_size * self.num_group, self.num_group)
            distances_batch[closest_point_idx] = float("inf")
            distances_batch = distances_batch.view(
                batch_size, self.num_group, self.num_group).transpose(1, 2).contiguous()
        sorted_indices = torch.stack(sorted_indices_list, dim=-1)
        sorted_indices = sorted_indices.view(-1)

        return sorted_indices


    def forward(self, x,xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        xyz = xyz.contiguous()
        # fps the centers out
        fps_idx = furthest_point_sample(xyz,self.num_group).long()  # B G 3
        center = index_points(xyz,fps_idx)
        new_points = index_points(x,fps_idx)
        # knn to get the neighborhood
        _,idx = self.knn(xyz, center)  # B G nsample
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
         # B,G,nsample 排序
        idx_base = torch.arange(
            0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = x.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, -1).contiguous() # B,G,nsample,3
        # normalize # B,N,C
        mean_x = new_points.unsqueeze(dim=-2)
        neighborhood = (neighborhood - mean_x)
        neighborhood = torch.cat([neighborhood,new_points.unsqueeze(2).repeat(1,1,self.group_size,1)],dim = -1)

        # can utilize morton_sorting by choosing morton_sorting function
        sorted_indices = self.simplied_morton_sorting(xyz, center)
        neighborhood = neighborhood.view(
            batch_size * self.num_group, self.group_size, -1)[sorted_indices, :, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, -1).contiguous()
        center = center.view(
            batch_size * self.num_group, -1)[sorted_indices, :]
        center = center.view(
            batch_size, self.num_group, -1).contiguous()

        # print(neighborhood.shape)
        return neighborhood, center


class PointNetv2(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.encoder1 = Encoder_small(in_channel,768)
        self.encoder2 = Encoder_large(768 * 2,768)
        self.group1 = Group(512, 32)
        self.group2 = Group(196,32)

    def forward(self, x):
        xyz = x[:,:,:3]
        # B,N,C
        l1_points,l1_xyz = self.group1(x,xyz) # B,G,M,12
        l1_points = self.encoder1(l1_points)
        l2_points,l2_xyz = self.group2(l1_points,l1_xyz)
        l2_points = self.encoder2(l2_points)
        # print(l2_points.shape)
        return l1_xyz,l1_points,l2_xyz,l2_points


if __name__=='__main__':
    x = torch.randn(2,2048,7).to('cuda')
    model = PointNetv2(in_channel=14).to('cuda')
    l1_xyz,l1_points,l2_xyz,l2_points = model(x)
    print(l1_xyz.shape,l1_points.shape,l2_xyz.shape,l2_points.shape)