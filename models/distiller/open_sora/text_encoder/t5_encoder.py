from colossalai.shardformer.modeling.jit import get_jit_fused_dropout_add_func
from colossalai.shardformer.modeling.t5 import get_jit_fused_T5_layer_ff_forward, get_T5_layer_self_attention_forward
from colossalai.shardformer.policies.base_policy import Policy, SubModuleReplacementDescription

import torch
import torch.nn as nn


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

    @staticmethod
    def from_native_module(module, *args, **kwargs):
        assert module.__class__.__name__ == "FusedRMSNorm", (
            "Recovering T5LayerNorm requires the original layer to be apex's Fused RMS Norm."
            "Apex's fused norm is automatically used by Hugging Face Transformers https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L265C5-L265C48"
        )

        layer_norm = T5LayerNorm(module.normalized_shape, eps=module.eps)
        layer_norm.weight.data.copy_(module.weight.data)
        layer_norm = layer_norm.to(module.weight.device)
        return layer_norm


class T5EncoderPolicy(Policy):
    def config_sanity_check(self):
        assert not self.shard_config.enable_tensor_parallelism
        assert not self.shard_config.enable_flash_attention

    def preprocess(self):
        return self.model

    def module_policy(self):
        from transformers.models.t5.modeling_t5 import T5LayerFF, T5LayerSelfAttention, T5Stack

        policy = {}

        # check whether apex is installed
        try:

            # recover hf from fused rms norm to T5 norm which is faster
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="layer_norm",
                    target_module=T5LayerNorm,
                ),
                policy=policy,
                target_key=T5LayerFF,
            )
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(suffix="layer_norm", target_module=T5LayerNorm),
                policy=policy,
                target_key=T5LayerSelfAttention,
            )
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(suffix="final_layer_norm", target_module=T5LayerNorm),
                policy=policy,
                target_key=T5Stack,
            )
        except (ImportError, ModuleNotFoundError):
            pass

        # use jit operator
        if self.shard_config.enable_jit_fused:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_T5_layer_ff_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=T5LayerFF,
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_T5_layer_self_attention_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=T5LayerSelfAttention,
            )

        return policy

    def postprocess(self):
        return self.model
