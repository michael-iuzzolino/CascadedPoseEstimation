import torch
from torch import nn
from models.hourglass import Conv, Hourglass, Pool, Residual
from models import tdl


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
        self._cascaded = kwargs.get("cascaded")
        print("self._cascaded: ", self._cascaded)
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
        self.n_stacks = n_stacks
        
        if self._cascaded:
            tdl_mode = kwargs.get("tdl_mode", "OSD")
            self.tdlines = [tdl.setup_tdl_kernel(tdl_mode, kwargs)
                            for _ in range(n_stacks-1)]
    
    def _reset_tdlines(self):
         [tdline.reset() for tdline in self.tdlines]
        
    @property
    def timesteps(self):
        return self.n_stacks
      
    def forward(self, x, t=None, is_train=True):
        if self._cascaded and t == 0:
            self._reset_tdlines()
            
        ## our posenet
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.n_stacks):
            if self._cascaded and i > t:
                break
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.n_stacks - 1:
                residual = self.merge_preds[i](preds) + self.merge_features[i](feature)
                if self._cascaded:
                    residual = self.tdlines[i](residual)
                try:
                    x = x + residual
                except:
                    print("\nx: ", x.shape)
                    print("residual: ", residual.shape)
                    exit()
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