import torch
import torch.nn as nn
from loguru import logger
import os
import sys


rank = int(os.environ.get('RANK', 0))


logger.remove()
logger.add(
    sys.stderr,
    filter=lambda record: rank in record["extra"].get("rank",[0]),
    format=(
        "<green>{time:MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<cyan>{file}:{line}</cyan> | "
        "{message}"
    ),
)

def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def split_params_by_module_type(model: nn.Module):
    """
    Split parameters using MODULE TYPES (not names).

    AdamW:
      - all biases
      - all normalization layers
      - input layer (conv1)
      - output layer (fc)

    Muon:
      - weights of hidden Conv2d / Linear layers
    """
    model = unwrap(model)

    muon_params = []
    muon_params_names = []
    adamw_params = []
    adamw_params_names = []
    no_param_modules = set()

    not_used = set()

    for module in model.modules():
        # ---- Normalization layers → AdamW
        if len(list(module.children())) != 0:
            continue
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.BatchNorm3d, nn.LayerNorm,
                                nn.GroupNorm)):
            count = 0
            for name, p in module.named_parameters(recurse=False):
                if p.requires_grad:
                    adamw_params.append(p)
                    adamw_params_names.append(name)
                else:
                    count += 1
            if count > 0:
                logger.debug(f'{module} has {count} params that require no grad')

        # ---- Convolution layers
        elif isinstance(module, nn.Conv2d):
            if module is model.conv1:
                # input layer → AdamW
                count = 0
                for name, p in module.named_parameters(recurse=False):
                    if p.requires_grad:
                        adamw_params.append(p)
                        adamw_params_names.append(name)
                    else:
                        count += 1
                if count > 0:
                    logger.debug(f'{module} has {count} params that require no grad')

            else:
                # hidden conv
                count = 0
                for name, p in module.named_parameters():
                    if p.requires_grad:
                        if p is module.weight:
                            muon_params.append(module.weight)
                            muon_params_names.append(name)
                        elif p is module.bias:
                            adamw_params.append(module.bias)
                            adamw_params_names.append(name)
                        else:
                            exit('error')
                    else:
                        count += 1
                if count > 0:
                    logger.debug(f'{module} has {count} params that require no grad')

        # ---- Linear layers
        elif isinstance(module, nn.Linear):
            if hasattr(model, 'fc') and module is model.fc:
                # output layer → AdamW
                count = 0
                for name, p in module.named_parameters(recurse=False):
                    if p.requires_grad:
                        adamw_params.append(p)
                        adamw_params_names.append(name)
                    else:
                        count += 1
                if count > 0:
                    logger.debug(f'{module} has {count} params that require no grad')


            else:
                # hidden linear
                count = 0
                for name, p in module.named_parameters():
                    if p.requires_grad:
                        if p is module.weight:
                            muon_params.append(module.weight)
                            muon_params_names.append(name)
                        elif p is module.bias:
                            adamw_params.append(module.bias)
                            adamw_params_names.append(name)
                        else:
                            exit('error')
                    else:
                        count += 1
                if count > 0:
                    logger.debug(f'{module} has {count} params that require no grad')

        else:
            for name, p in module.named_parameters():
                logger.debug(f'missing param {name} in {module}')
            no_param_modules.add(type(module))
            

    # Safety: remove duplicates and preserve order， if using list(set()) then order is broken
    muon_len = len(muon_params)
    adamw_len = len(adamw_params)
    muon_params = list(dict.fromkeys(muon_params))
    adamw_params = list(dict.fromkeys(adamw_params))
    muon_params_names = list(dict.fromkeys(muon_params_names))
    adamw_params_names = list(dict.fromkeys(adamw_params_names))
    logger.debug(f'no_param_modules, {no_param_modules}')

    return muon_params, adamw_params, muon_params_names, adamw_params_names
