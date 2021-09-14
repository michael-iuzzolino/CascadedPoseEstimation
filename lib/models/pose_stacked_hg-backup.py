import torch
from torch import nn
from models import tdl

Pool = nn.MaxPool2d


def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)
  

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
    
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        
        self.need_skip = inp_dim != out_dim
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 
      

class Hourglass(nn.Module):
    def __init__(self, layer_i, n_planes, bn=None, increase=0):
        super(Hourglass, self).__init__()
        self.layer_i = layer_i
        self.pool = Pool(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        # Upper branch
        self.upper_res_1 = Residual(inp_dim=n_planes, out_dim=n_planes)
        
        # Lower branch
        n_out_planes = n_planes + increase
        self.lower_res_1 = Residual(inp_dim=n_planes, out_dim=n_out_planes)
        
        # Recursive hourglass
        if self.layer_i > 1:
            self.low = Hourglass(
                layer_i=layer_i-1,
                n_planes=n_out_planes,
                bn=bn,
            )
        else:
            self.low = Residual(inp_dim=n_out_planes, out_dim=n_out_planes)
        self.lower_res_2 = Residual(inp_dim=n_out_planes, out_dim=n_planes)
        
    def forward(self, x):
        identity  = self.upper_res_1(x)
        out = self.pool(x)
        out = self.lower_res_1(out)
        out = self.low(out)
        out = self.lower_res_2(out)
        residual  = self.upsample(out)
        return identity + residual
      

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

    
class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):
    def __init__(self, n_stacks, inp_dim, n_joints, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        self.n_stacks = n_stacks
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(n_stacks)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(n_stacks)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, n_joints, 1, relu=False, bn=False) 
                                    for i in range(n_stacks)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) 
                                              for i in range(n_stacks-1)] )
        self.merge_preds = nn.ModuleList( [Merge(n_joints, inp_dim) 
                                           for i in range(n_stacks-1)] )
      
    def forward(self, x, t=None, is_train=True):
        ## our posenet
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.n_stacks):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.n_stacks - 1:
                residual = self.merge_preds[i](preds) + self.merge_features[i](feature)
                x = x + residual
                
        outs = torch.stack(combined_hm_preds, 1)
        if is_train:
            outs = outs[:,-1]
        return outs
      

def get_pose_net(cfg, is_train, **kwargs):
  n_hg_stacks = cfg.MODEL.EXTRA.N_HG_STACKS
  model = PoseNet(n_stacks=n_hg_stacks,
                  inp_dim=cfg.MODEL.IMAGE_SIZE[0],
                  n_joints=cfg.MODEL.NUM_JOINTS,
                  cascaded=cfg.MODEL.CASCADED,
                  tdl_mode=cfg.MODEL.EXTRA.TDL_MODE,
                  tdl_alpha=cfg.MODEL.EXTRA.TDL_ALPHA,)

#     if is_train and cfg.MODEL.INIT_WEIGHTS:
#         model.init_weights(cfg.MODEL.PRETRAINED)

  return model