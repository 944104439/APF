
from pointnet2_ops.pointnet2_utils import *


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
    idx = idx.long()
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=True):
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
    xyz = xyz.contiguous()
    points = points.contiguous()

    fps_idx = furthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = ball_query(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]

    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_xyz_norm = grouped_xyz_norm/radius

    # 用一个sort
    # sortidx = simplied_morton_sorting(new_xyz)
    # grouped_xyz_norm = grouped_xyz_norm.reshape(B*npoint,nsample,-1)[sortidx,:]
    # grouped_xyz_norm = grouped_xyz_norm.reshape(B,npoint,nsample,-1).contiguous()


    if points is not None:
        grouped_points = index_points(points, idx)
        # grouped_points = grouped_points.reshape(B * npoint, nsample, -1)[sortidx, :]
        # grouped_points = grouped_points.reshape(B, npoint, nsample, -1).contiguous()
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


class SABlock(nn.Module):
    """
    Set abstraction block without downsampling.
    """
    def __init__(self, in_dim=32, out_dim=64, stride=1, layers=3, radius=0.1, k=16):
        super().__init__()
        self.stride = stride
        self.radius = radius
        self.layers = layers
        self.k = k
        # *是重复

        # print(f"self.stride = {self.stride}")
        dims = [in_dim + 3] + [out_dim] * layers

        if layers == 1:
            self.convs = nn.Conv2d(dims[0], dims[1], 1, bias=False)
            self.norm = nn.BatchNorm1d(out_dim)
            self.act = nn.ReLU()
        else:
            self.skip_conv = nn.Conv1d(in_dim, out_dim, 1, bias=False) if in_dim != out_dim else nn.Identity()
            self.convs = nn.Sequential(*[
                nn.Sequential(nn.Conv2d(in_d, out_d, 1, bias=False),
                              nn.BatchNorm2d(out_d),
                              nn.ReLU())
                for in_d, out_d in zip(dims[:-2], dims[1:-1])
            ])
            self.convs.append(nn.Conv2d(dims[-2], dims[-1], 1, bias=False))
            self.norm = nn.BatchNorm1d(out_dim)
            self.act = nn.ReLU()

    def forward(self, x, xyz):

        """
          input : X -> B,C,N
                xyz -> B,3,n
                output -> B,out_dim,N ,B,3,N
        """
        x = x.permute(0,2,1)
        xyz = xyz.permute(0,2,1)
        # B,N,C, B,N,3
        B,N,C = x.shape
        # B,npoint,nsample,3+d
        if self.stride == 1:
            samples = 196
        else:
            samples = N//self.stride
        new_xyz, new_points, grouped_xyz, fps_idx = sample_and_group(npoint=samples,radius=self.radius,nsample=self.k,xyz=xyz,points=x)
        new_points = new_points.permute(0,3,1,2)
        # x中的原有的points
        inputs = index_points(x,fps_idx).permute(0,2,1) # B,C,N
        x = self.convs(new_points)
        x = x.max(dim=-1)[0] # B,C,npoint
        if hasattr(self, 'skip_conv'):
            x = self.skip_conv(inputs) + x
        x = self.act(self.norm(x))

        new_xyz = new_xyz.permute(0, 2, 1)

        return x,new_xyz


class InvResMLP(nn.Module):

    def __init__(self, in_dim =64, expansion=4, radius=0.1, k=32):
        super().__init__()
        self.sa_conv = SABlock(in_dim, in_dim, stride=1, layers=1, radius=radius, k=k)

        dims = [in_dim, in_dim * expansion, in_dim]
        self.conv = nn.Sequential(
            nn.Conv1d(dims[0], dims[1], 1, bias=False),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(),
            nn.Conv1d(dims[1], dims[2], 1, bias=False),
            nn.BatchNorm1d(dims[2])
        )
        self.act = nn.ReLU()

    def forward(self, x, xyz):
        inputs = x
        x,_ = self.sa_conv(x, xyz)
        x = self.conv(x)
        x = self.act(inputs + x)

        return x


class PointNextEncoder(nn.Module):

    def __init__(
            self,
            in_dim=7,
            dims=[32, 64, 128, 256, 512],  # dims[0] is the dim of the stem output
            blocks=[4, 7, 4, 4],  # blocks: sa + invres
            strides=[2, 2, 2, 1],
            radius=0.15,
            k=32,
            sa_layers=1,
    ):
        """
        :param in_dim: the dim of input features
        :param dims: dims of each stage
        :param blocks: number of blocks in each stage
        :param strides: strides of each stage
        :param radius: the first radius of sa layer
        :param k: k at each stage
        :param sa_layers: number of sa layers in each stage
        """
        super().__init__()
        self.encoder_dims = dims

        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, dims[0], 1, bias=False),
            nn.BatchNorm1d(dims[0]),
            nn.ReLU()
        )

        radius_scaling = 1.5
        # 0.1 0.2 0.4 0.8
        radii = [radius * (radius_scaling ** i) for i in range(len(blocks))]
        print(radii)
        self.encoder = nn.ModuleList()

        for i in range(len(blocks)):
            layers = nn.Sequential(
                SABlock(dims[i], dims[i + 1], stride=strides[i], layers=sa_layers, radius=radii[i], k=k),
                *[InvResMLP(dims[i + 1], radius=radii[i] * radius_scaling, k=k) for _ in range(blocks[i] - 1)]
            )

            self.encoder.append(layers)

        self.out_dim = dims[-1]

    def forward_features(self, x, xyz):

        xyz = xyz.permute(0,2,1)
        x = self.stem(x.permute(0,2,1)) # 2,32,1024
        features = [(x, xyz)] # x,xyz

        for block in self.encoder:
            # SA module  input -> B,C,N return-> B,3,N, B,C,N
            x,xyz = block[0](x, xyz)
            # InvRes
            for layer in block[1:]:
                x = layer(x, xyz)
            features.append((x, xyz))

        return features

    def forward(self, x, xyz):
        return self.forward_features(x, xyz)


def pointnext_s(**kwargs):
    model_kwargs = dict(blocks=[1, 1, 1, 1], sa_layers=2, **kwargs)
    return PointNextEncoder(**model_kwargs)


def pointnext_b(**kwargs):
    model_kwargs = dict(blocks=[2, 3, 2, 2], sa_layers=1, **kwargs)
    return PointNextEncoder(**model_kwargs)


def pointnext_l(**kwargs):
    model_kwargs = dict(blocks=[3, 5, 3, 3], sa_layers=1, **kwargs)
    return PointNextEncoder(**model_kwargs)


def pointnext_xl(**kwargs):
    model_kwargs = dict(dims=[64, 128, 256, 512, 1024], blocks=[4, 7, 4, 4], sa_layers=1, **kwargs)
    # print(model_kwargs)
    return PointNextEncoder(**model_kwargs)


class PointNext(nn.Module):
    def __init__(self, feat_dim = 512,out_dim = 40):
        super().__init__()
        self.encoder = pointnext_s()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256,out_dim)
        )
    def forward(self, x):
        xyz = x[:,:,:3]
        feats = self.encoder(x, xyz)
        out = feats[-1][0]
        out = torch.max(out,dim = -1)[0]
        out = self.classifier(out)
        return out


class PointNextCls(nn.Module):
    def __init__(self, feat_dim = 512,out_dim = 15):
        super().__init__()
        self.encoder = pointnext_s()
        self.tokenize = nn.Linear(feat_dim,768)

    def forward(self, x):
        xyz = x[:,:,:3]
        feats = self.encoder(x, xyz)
        out,xyz= feats[-1][0],feats[-1][1]
        out = self.tokenize(out.permute(0,2,1))
        return out,feats


def pointNext():
    return PointNextCls()


if __name__ == '__main__':
    # PointNextEncoder()
    x = torch.randn(2,2048,7).to('cuda')
    xyz = x[:,:,:3]
    model = PointNext().to('cuda')
    out= model(x)
    print(out.shape)


