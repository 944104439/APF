from PromptModels.pointnet import PointNet
from PromptModels.pointnetv2 import PointNetv2
from PromptModels.utils import *
# from PromptModels.pointnet import *


class APF(nn.Module):

    def __init__(self,num_classes=40, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,act_layer=nn.GELU, npoint=196, nsample=32, normal=False, in_channel=3):
        # Recreate ViT
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.embed = embed_dim
        self.enc_norm = norm_layer(embed_dim)
        self.npoint = npoint
        self.nsample = nsample
        self.normal = normal

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        self.num_classes = num_classes
        self.in_channel = in_channel
        # self.encoder = PointNetv2(in_channel=8)

        self.encoder = PointNet(embed_dim, npoint, nsample, self.in_channel)
        self.head = PointCls(in_channel=self.embed, out_channel=self.num_classes)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

        for name, param in self.named_parameters():
            if 'adaptmlp' in name or 'head' in name or 'enc_norm' in name or 'encoder' in name:
                param.requires_grad = True

    def forward_features(self, x):
        B,N,_ = x.shape
        x = self.encoder(x)
        # _,_,_,x = self.encoder(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x,temp=1)

        x = self.enc_norm(x)  # b,n,c
        x_pr = x.max(-2)[0]
        return x_pr

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.head(x)
        return x


def pointFormer(num_classes=15, base_model='vit_base_patch16_224_in21k', npoint=196, nsample=32, in_channel=3):
    print("===> create model")
    import timm
    basic_model = timm.create_model(base_model, pretrained=True)
    base_state_dict = basic_model.state_dict()
    del base_state_dict['head.weight']
    del base_state_dict['head.bias']
    model = APF(num_classes=num_classes,depth=12, num_heads=12, embed_dim=768, npoint=npoint, nsample=nsample, in_channel=in_channel)
    model.load_state_dict(base_state_dict, False)
    model.freeze()
    return model


if __name__ == '__main__':

    x = torch.rand(2, 1568, 4).to('cuda')
    model = pointFormer(num_classes=15, npoint=196, nsample=32, in_channel=4*2).to('cuda')
    x = model(x)
    print(x.shape)
