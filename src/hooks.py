from nesymres.architectures.model import Model
import torch
from typing import Literal, Callable, Tuple, Union
from dataclasses import dataclass

@dataclass
class HookPoint:
    layer: int
    # Which part of the decoder layer to hook into. Can either be the MLP, a self-attention or cross-attention head.
    component: Union[Literal["mlp"], Tuple[Literal["self", "cross"], int]]

def register_decoder_hook(model: Model, hook_fn: Callable, hook: HookPoint) -> torch.utils.hooks.RemovableHandle:
    """
    Hooks a function into the decoder part of the model. This allows for reading or manulipulating the output of a specific component.

    NOTE: To remove the hook, the returned `RemovableHandle` must be called with `.remove()`.

    # Args
    * `model`: The model to hook into.
    * `hook_fn`: Callable that takes the output of the hooked component as a`torch.Tensor` and the hooked location as a `HookPoint`.
        The function should return an updated output.
    * `hook`: Description of the component to hook into.
    """

    def hook_wrapper(_module, _input, output):
        if hook.component == "mlp":
            output[0] = hook_fn(output[0], hook)
        elif hook.component[0] == "self" or hook.component[0] == "cross":
            head_idx = hook.component[1]

            # multihead and self-attention layer have same number of heads
            num_head = model.decoder_transfomer.layers[hook.layer].multihead_attn.num_heads
            
            # view data in terms of [seq_len x batch_size x num_head x head_dim])
            seq_len, bsz, _ = output[0].size()
            output_heads = output[0].view(seq_len, bsz, num_head, -1)

            # hook output of specified head
            output_heads[:, :, head_idx, :] = hook_fn(output_heads[:, :, head_idx, :], hook)
        else:
            raise ValueError(f"Unknown hook component: {hook.component}")
        
        return output
    
    if hook.component == "mlp":
        # hook into 2nd linear layer of MLP
        return model.decoder_transfomer.layers[hook.layer].linear2.register_forward_hook(hook_wrapper)
    elif hook.component[0] == "self":
        # hook into self-attention layer
        return model.decoder_transfomer.layers[hook.layer].self_attn.register_forward_hook(hook_wrapper)
    elif hook.component[0] == "cross":
        # hook into cross-attention layer
        return model.decoder_transfomer.layers[hook.layer].multihead_attn.register_forward_hook(hook_wrapper)