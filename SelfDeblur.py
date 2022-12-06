from hypnettorch.mnets.mnet_interface import MainNetInterface
from networks.skip import skip
from networks.fcn import fcn
import torch.nn as nn
import torch
from hypnettorch.utils.batchnorm_layer import BatchNormLayer

class SelfDeblurImage(MainNetInterface, nn.Module):
    def __init__(self, input_depth, pad, dtype, need_bias=True, need_sigmoid=True) -> None:
        super().__init__()

        net = skip( input_depth, 1,
                    num_channels_down = [128, 128, 128, 128, 128],
                    num_channels_up   = [128, 128, 128, 128, 128],
                    num_channels_skip = [16, 16, 16, 16, 16],
                    upsample_mode='bilinear',
                    need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
        self.model = net.type(dtype)
    
        self._replace_batch_norm_layers()
        self._layer_weight_tensors = [module.weight for module in self.model.modules() if hasattr(module, "weight")]
        self._layer_bias_vectors = [module.bias for module in self.model.modules() if hasattr(module, "weight")]
        self._internal_params = nn.ParameterList(self._layer_weight_tensors + self._layer_bias_vectors)
        self._layer_weight_tensors = nn.ParameterList(self._layer_weight_tensors)
        self._layer_bias_vectors = nn.ParameterList(self._layer_bias_vectors)
        self._param_shapes = [param.shape for param in self._internal_params]
        self._hyper_shapes_learned = self._param_shapes
        self._hyper_shapes_distilled = None
        self._has_bias = need_bias
        self._has_fc_out = False
        self._mask_fc_out = False
        self._has_linear_out = not need_sigmoid

    def _replace_batch_norm_layers(self):
        self._batchnorm_layers = []
        for module in self.model.modules():
            if isinstance(module, nn.Sequential):
                for i in range(len(module)):
                    if isinstance(module[i], nn.BatchNorm2d):
                        num_features = module[i].num_features
                        momentum = module[i].momentum
                        affine = module[i].affine
                        track_running_stats = module[i].track_running_stats
                        module[i] = BatchNormLayer(num_features=num_features, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
                        self._batchnorm_layers.append(module[i])
        self._batchnorm_layers = nn.ModuleList(self._batchnorm_layers)

    def forward(self, x: tuple, weights=None, distilled_params=None, condition=None):
        i = 0
        if weights != None:
            for module in self.model.modules():
                if hasattr(module, "weight"):
                    module.weight = nn.Parameter(weights[i], requires_grad=True)
                    module.bias = nn.Parameter(weights[len(self.layer_weight_tensors) + i], requires_grad=True)
                    i += 1
        else:
            return self.model(x)

    def distillation_targets(self):
        return []


class SelfDeblurKernel(MainNetInterface, nn.Module):
    def __init__(self, dtype, n_k, kernel_shape: tuple, need_bias=True, need_sigmoid=True) -> None:
        super().__init__()

        net_kernel = fcn(n_k, kernel_shape[0]*kernel_shape[1])
        self.model = net_kernel.type(dtype)

        self._replace_batch_norm_layers()
        self._layer_weight_tensors = [module.weight for module in self.model.modules() if hasattr(module, "weight")]
        self._layer_bias_vectors = [module.bias for module in self.model.modules() if hasattr(module, "weight")]
        self._internal_params = nn.ParameterList(self._layer_weight_tensors + self._layer_bias_vectors)
        self._layer_weight_tensors = nn.ParameterList(self._layer_weight_tensors)
        self._layer_bias_vectors = nn.ParameterList(self._layer_bias_vectors)
        self._param_shapes = [param.shape for param in self._internal_params]
        self._hyper_shapes_learned = self._param_shapes
        self._hyper_shapes_distilled = None
        self._has_bias = need_bias
        self._has_fc_out = False
        self._mask_fc_out = False
        self._has_linear_out = not need_sigmoid

    def _replace_batch_norm_layers(self):
        self._batchnorm_layers = []
        for module in self.model.modules():
            if isinstance(module, nn.Sequential):
                for i in range(len(module)):
                    if isinstance(module[i], nn.BatchNorm2d):
                        num_features = module[i].num_features
                        momentum = module[i].momentum
                        affine = module[i].affine
                        track_running_stats = module[i].track_running_stats
                        module[i] = BatchNormLayer(num_features=num_features, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
                        self._batchnorm_layers.append(module[i])
        self._batchnorm_layers = nn.ModuleList(self._batchnorm_layers)

    def forward(self, x: tuple, weights=None, distilled_params=None, condition=None):
        i = 0
        if weights != None:
            for module in self.model.modules():
                if hasattr(module, "weight"):
                    module.weight = nn.Parameter(weights[i], requires_grad=True)
                    module.bias = nn.Parameter(weights[len(self.layer_weight_tensors) + i], requires_grad=True)
                    i += 1
        else:
            return self.model(x)

    def distillation_targets(self):
        return []