"""Explicit, reproducible grayscale segmentation architectures; no weight downloads."""

import copy
import importlib.metadata


MODEL_IDS = tuple(
    [f'unet_{width}' for width in (16, 32, 64)]
    + [f'{family}_resnet{depth}' for family in ('unetpp', 'fpn', 'deeplabv3plus')
       for depth in (18, 34, 50)]
    + [f'segformer_b{size}' for size in (0, 1, 2)]
)


def model_spec(model_id):
    if model_id not in MODEL_IDS:
        raise ValueError(f'Unknown model: {model_id}. Choose from {MODEL_IDS}')
    family, size = model_id.split('_', 1)
    spec = dict(id=model_id, family=family, encoder_weights=None, in_channels=1, classes=1,
                activation=None, normalization='GroupNorm (BatchNorm replaced; max 8 groups)')
    if family == 'unet':
        spec.update(base_channels=int(size), implementation='local_grayscale_unet')
    else:
        spec.update(encoder_name=f'mit_{size}' if family == 'segformer' else size,
                    implementation='segmentation_models_pytorch', implementation_version='0.5.0')
        spec['decoder'] = {
            'unetpp': dict(decoder_channels=[256, 128, 64, 32, 16]),
            'fpn': dict(decoder_pyramid_channels=256, decoder_segmentation_channels=128),
            'deeplabv3plus': dict(encoder_output_stride=16, decoder_channels=256),
            'segformer': dict(decoder_segmentation_channels=256),
        }[family]
    return spec


def replace_batch_norm(module):
    """GroupNorm avoids batch-size-dependent statistics and 1x1 pooling failures."""
    from torch import nn
    for name, child in list(module.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            groups = min(8, child.num_features)
            while child.num_features % groups:
                groups -= 1
            replacement = nn.GroupNorm(groups, child.num_features, eps=child.eps, affine=child.affine)
            if child.affine:
                replacement.weight.data.copy_(child.weight.data)
                replacement.bias.data.copy_(child.bias.data)
            setattr(module, name, replacement)
        else:
            replace_batch_norm(child)


def build_model(spec, activation_checkpointing=False):
    import torch
    from torch import nn
    spec = copy.deepcopy(spec)
    if spec != model_spec(spec['id']):
        raise ValueError('Unsupported or altered architecture specification.')
    if spec['family'] == 'unet':
        from .model import UNet
        return UNet(spec['base_channels'], activation_checkpointing)
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise RuntimeError('Install benchmark dependencies: uv sync --extra confident --extra segmentation-benchmark') from error
    if importlib.metadata.version('segmentation-models-pytorch') != spec['implementation_version']:
        raise RuntimeError('This benchmark requires segmentation-models-pytorch==0.5.0.')
    constructor = {'unetpp': smp.UnetPlusPlus, 'fpn': smp.FPN,
                   'deeplabv3plus': smp.DeepLabV3Plus, 'segformer': smp.Segformer}[spec['family']]
    net = constructor(encoder_name=spec['encoder_name'], encoder_weights=None,
                      in_channels=1, classes=1, activation=None, **spec['decoder'])
    replace_batch_norm(net)

    class CheckpointedModel(nn.Module):
        # Keep the same state_dict keys for fitting and inference.
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            if activation_checkpointing and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                return checkpoint(self.model, x, use_reentrant=False)
            return self.model(x)

    return CheckpointedModel(net)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())
