import torch
import torch.nn as nn
import torch.nn.functional as F


class OneStepDelayKernel(nn.Module):
  """Single slot queue OSD kernel."""

  def __init__(self, *args, **kwargs):
    """Initialize OSD kernel."""
    super().__init__()
    self.reset()

  def reset(self):
    self.state = None

  def forward(self, current_state):
    if self.state is not None:
      prev_state = self.state
    else:
      prev_state = torch.zeros_like(current_state)
      prev_state.requires_grad = True

    self.state = current_state.clone()

    return prev_state
  
  
class BottleneckBlock(nn.Module):
  def __init__(self, inplanes, outplanes, stride=1, downsample=False, cascaded=False):
    super().__init__()
    halfplanes = outplanes // 2
    self._downsample = downsample
    self._cascaded = cascaded
    
    self.relu = nn.ReLU()
    self.identity_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
    
    self.bn1 = nn.BatchNorm2d(inplanes)
    self.conv1 = nn.Conv2d(inplanes, halfplanes, kernel_size=1, padding=0)
    
    self.bn2 = nn.BatchNorm2d(halfplanes)
    self.conv2 = nn.Conv2d(halfplanes, halfplanes, kernel_size=3, stride=stride, padding=1)
    
    self.bn3 = nn.BatchNorm2d(halfplanes)
    self.conv3 = nn.Conv2d(halfplanes, outplanes, kernel_size=1, padding=0)
    
    if self._cascaded:
      self.tdline = OneStepDelayKernel()
    
  def set_time(self, t):
    self.t = t
    if t == 0:
      self.tdline.reset()
      
  def forward(self, inputs):
    identity = inputs
    if self._downsample:
      identity = self.identity_conv(inputs)
    
    out = self.bn1(inputs)
    out = self.relu(out)
    out = self.conv1(out)
    
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv2(out)
    
    out = self.bn3(out)
    out = self.relu(out)
    residual = self.conv3(out)
    
    if self._cascaded:
      residual = self.tdline(residual)
    
    out = residual + identity
    
    return out
  
  
class DoubleConv(nn.Module):
  """(convolution => [BN] => ReLU) * 2"""

  def __init__(self, in_channels, out_channels, mid_channels=None):
    super().__init__()
    if not mid_channels:
        mid_channels = out_channels
    self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)


class Down(nn.Module):
  """Downscaling with maxpool then double conv"""

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.maxpool_conv = nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConv(in_channels, out_channels)
    )

  def forward(self, x):
    return self.maxpool_conv(x)


class Up(nn.Module):
  """Upscaling then double conv"""

  def __init__(self, in_channels, out_channels, bilinear=True):
    super().__init__()

    # if bilinear, use the normal convolutions to reduce the number of channels
    if bilinear:
      self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
      self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    else:
      self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
      self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)


class OutConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(OutConv, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  def forward(self, x):
    return self.conv(x)
  

class HourGlassModule(nn.Module):
  """
  https://github.com/milesial/Pytorch-UNet/tree/master/unet
  """
  def __init__(self, n_channels, n_classes, bilinear=True, cascaded=False):
    super().__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.bilinear = bilinear
    self._cascaded = cascaded
    
    filters = [32, 64, 128, 256, 512]

    self.inc = DoubleConv(n_channels, filters[0])
    self.bottleneck_block_1 = BottleneckBlock(filters[0], filters[1], downsample=True, cascaded=self._cascaded)
    self.bottleneck_block_2 = BottleneckBlock(filters[1], filters[2], downsample=True, cascaded=self._cascaded)
    self.bottleneck_block_3 = BottleneckBlock(filters[2], filters[3], downsample=True, cascaded=self._cascaded)
    
    factor = 2 if bilinear else 1
    self.down4 = Down(filters[3], filters[4] // factor)
    self.up1 = Up(filters[4], filters[3] // factor, bilinear)
    self.up2 = Up(filters[3], filters[2] // factor, bilinear)
    self.up3 = Up(filters[2], filters[1] // factor, bilinear)
    self.up4 = Up(filters[1], filters[0], bilinear)
    self.outc = OutConv(filters[0], n_classes)

  def forward(self, x):
    x1 = self.inc(x)
    x2 = self.bottleneck_block_1(x1)
    x3 = self.bottleneck_block_2(x2)
    x4 = self.bottleneck_block_3(x3)
    x5 = self.down4(x4)
    
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    logits = self.outc(x)
    return logits

  
class LinearLayer(nn.Module):
  def __init__(self, inplanes, outplanes):
    super().__init__()
    self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
    self.bn = nn.BatchNorm2d(outplanes)
    self.relu = nn.ReLU()
  
  def forward(self, x):
    out = self.conv(x)
    out = self.bn(out)
    out = self.relu(out)
    return out
  
  
class HeadLayer(nn.Module):
  def __init__(self, out_channels=256, cascaded=False):
    super().__init__()
    self.relu = nn.ReLU()
    self._cascaded = cascaded
    
    filters = [32, 64, 128, out_channels]
    
    self.conv1 = nn.Conv2d(3, filters[0], kernel_size=7, stride=2, padding=3)
    self.bn1 = nn.BatchNorm2d(filters[0])
    self.mp1 = nn.MaxPool2d(2)
    
    self.bottleneck_block_1 = BottleneckBlock(filters[0], filters[1], downsample=True, cascaded=self._cascaded)
    self.bottleneck_block_2 = BottleneckBlock(filters[1], filters[2], downsample=True, cascaded=self._cascaded)
    self.bottleneck_block_3 = BottleneckBlock(filters[2], filters[3], downsample=True, cascaded=self._cascaded)
  
  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.mp1(out)
    out = self.relu(out)
    out = self.bottleneck_block_1(out)
    out = self.bottleneck_block_2(out)
    out = self.bottleneck_block_3(out)
    return out
  

class CascadedUNet(nn.Module):
  def __init__(self, n_stacks=1, n_joints=16, dim=64, cascaded=False):
    super().__init__()
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    
    self._n_stacks = n_stacks
    self._n_joints = n_joints
    self._cascaded = cascaded
    self._dim = dim
    
    self.head_layer = HeadLayer(out_channels=dim, cascaded=self._cascaded)
  
    self.hourglass_modules = []
    for _ in range(self._n_stacks):
      hourglass_i = HourGlassModule(dim, dim, cascaded=self._cascaded)
      self.hourglass_modules.append(hourglass_i)
    
    # Add modules
    for stack_i, hourglass_i in enumerate(self.hourglass_modules):
      self.add_module(f"hourglass_{stack_i}", hourglass_i)
    
    self.bottleneck_block_1 = BottleneckBlock(dim, dim, 
                                              downsample=False, 
                                              cascaded=self._cascaded)
    self.linear_layer = LinearLayer(dim, dim)
    self.out_conv = nn.Conv2d(dim, n_joints, kernel_size=1)
    
    if self._n_stacks > 1:
      self.inter_conv_1 = nn.Conv2d(dim, dim, kernel_size=1)
      self.inter_conv_2 = nn.Conv2d(n_joints, dim, kernel_size=1)
    
    # Compute n_timesteps
    self._compute_n_timesteps()
  
  def _compute_n_timesteps(self):
    n_timesteps = 0
    for key, _ in self.named_modules():
      if 'bottleneck_block' in key:
        bn_key = key.split('bottleneck_block_')[1]
        if '.' in bn_key:
          continue
        n_timesteps += 1
    self.n_timesteps = n_timesteps + 1
    
  def _set_time(self, t):
    for key, module in self.named_modules():
      if 'bottleneck_block' in key:
        bn_key = key.split('bottleneck_block_')[1]
        if '.' in bn_key:
          continue
        module.set_time(t)
        
  def _forward(self, x, t):
    # Set time on all blocks
    if self._cascaded:
      self._set_time(t)
    
    # initial processing of the image
    out = self.head_layer(x)
    
    ys = []
    for stack_i, hourglass_mod in enumerate(self.hourglass_modules):
      out = hourglass_mod(out)
      out = self.bottleneck_block_1(out)
      out = self.linear_layer(out)
      out = self.out_conv(out)
      y = self.sigmoid(out)
      
      ys.append(y)
      
      if stack_i < self._n_stacks - 1:
        y_inter_1 = self.inter_conv_1(out)
        y_inter_2 = self.inter_conv_2(y)
        out = y_inter_1 + y_inter_2
        
    return ys
    
  def forward(self, x, t=None):
    if t is not None:
      return self._forward(x, t)[-1]
    elif not self._cascaded:
      return self._forward(x, t=0)[-1]
    else:
      for t in range(self.n_timesteps):
        out = self._forward(x, t)[-1]
      return out


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    model = CascadedUNet(n_stacks=1, 
                         n_joints=cfg.MODEL.NUM_JOINTS,
                         cascaded=cfg.MODEL.CASCADED)

#     if is_train and cfg.MODEL.INIT_WEIGHTS:
#         model.init_weights(cfg.MODEL.PRETRAINED)

    return model