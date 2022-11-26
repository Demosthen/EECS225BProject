import hypnettorch as hnet
from hypnettorch.mnets.mnet_interface import MainNetInterface
import torch
import torch.nn as nn
from networks.skip import skip
from hypnettorch.utils.batchnorm_layer import BatchNormLayer

class HyperDip(MainNetInterface, nn.Module):
    def __init__(self, num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True) -> None:
        super().__init__()
        self.model = skip(num_input_channels, num_output_channels, 
            num_channels_down, num_channels_up, num_channels_skip, 
            filter_size_down, filter_size_up, filter_skip_size,
            need_sigmoid, need_bias, pad, upsample_mode, downsample_mode, act_fun, 
            need1x1_up)
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
        self._has_linear_out = need_sigmoid
        
        
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

    def forward(self, x, weights=None, distilled_params=None, condition=None):
        print(len(weights), len(self.layer_weight_tensors), len(self.layer_bias_vectors))
        i = 0
        for module in self.model.modules():
            if hasattr(module, "weight"):
                module.weight = nn.Parameter(weights[i], requires_grad=True)
                module.bias = nn.Parameter(weights[len(self.layer_weight_tensors) + i], requires_grad=True)
                i += 1
        return self.model(x)

    def distillation_targets(self):
        return []