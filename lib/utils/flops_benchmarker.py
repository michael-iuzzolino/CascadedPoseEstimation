"""
Source: https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/d5df5e066fe9c6078d38b26527d93436bf869b1c/pytorch_segmentation_detection/utils/flops_benchmark.py
"""
import numpy as np
import torch


def hook_fn(m, i, o):
    if not isinstance(m, torch.nn.Conv2d):
        return
    input_tensor = i[0] if isinstance(i, tuple) else i
    output_tensor = o[0] if isinstance(o, tuple) else o
    kernel_h, kernel_w = m.kernel_size

    batch_size, in_channels, _, _ = np.array(input_tensor.size())
    _, out_channels, out_h, out_w = np.array(output_tensor.size())
    assert in_channels == m.in_channels, "In channels mismatch!"
    assert out_channels == m.out_channels, "In channels mismatch!"

    flops_per_position = 2 * in_channels * out_channels * kernel_h * kernel_w

    n_conv_flops = flops_per_position * batch_size * out_h * out_w

    n_bias_flops = 0
    if m.bias is not None:
        n_bias_flops = out_h * out_w * out_channels * batch_size

    n_module_flops = n_conv_flops + n_bias_flops
    m.__flops__ = n_module_flops
    
  
def init(net):
    net.compute_total_flops = compute_total_flops.__get__(net)
    net.cleanup_flop_hooks = cleanup_flop_hooks.__get__(net)
    add_flop_tracking(net)
  
  
def add_flop_tracking(module, recursion_i=0):
    if recursion_i > 20:
        return
    for key, layer in module.named_modules():
        if isinstance(layer, torch.nn.Sequential):
            add_flop_tracking(layer, recursion_i=recursion_i+1)
        else:
            # it"s a non sequential. Register a hook
            handle = layer.register_forward_hook(hook_fn)
            layer.__flops_handle__ = handle
        

def cleanup_flop_hooks(self):
    for k, m in self.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if hasattr(m, "__flops_handle__"):
               m.__flops_handle__.remove()
        

def compute_total_flops(self):
    n_flops = 0
    for _, m in self.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if hasattr(m, "__flops__"):
                n_flops += m.__flops__
    
    n_total_flops = n_flops / 1e9 / 2
    return n_total_flops