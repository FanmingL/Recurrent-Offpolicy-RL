import torch
from flash_attn.models.gpt_rl import GPTModel, GPT2Config
from torch.nn import Module
from flash_attn.utils.generation import InferenceParams


# GPT2Config={
#   "activation_function": "gelu_new",
#   "attn_pdrop": 0.1,
#   "bos_token_id": 50256,
#   "embd_pdrop": 0.1,
#   "eos_token_id": 50256,
#   "initializer_range": 0.02,
#   "layer_norm_epsilon": 1e-05,
#   "model_type": "gpt2",
#   "n_embd": 768,
#   "n_head": 12,
#   "n_inner": null,
#   "n_layer": 12,
#   "n_positions": 1024,
#   "reorder_and_upcast_attn": false,
#   "resid_pdrop": 0.1,
#   "scale_attn_by_inverse_layer_idx": false,
#   "scale_attn_weights": true,
#   "summary_activation": null,
#   "summary_first_dropout": 0.1,
#   "summary_proj_to_labels": true,
#   "summary_type": "cls_index",
#   "summary_use_proj": true,
#   "transformers_version": "4.39.3",
#   "use_cache": true,
#   "vocab_size": 50257
# }


class GPTLayer(Module):
    def __init__(self, ndim=768, nhead=12, nlayer=12, pdrop=0.1, norm_epsilon=1e-5):
        super().__init__()
        config = GPT2Config(n_embd=ndim, n_head=nhead, nlayer=nlayer, attn_pdrop=pdrop,
                            layer_norm_epsilon=norm_epsilon,
                            resid_pdrop=pdrop, embd_pdrop=pdrop,
                            )

        config.use_flash_attn = True
        config.fused_bias_fc = True
        config.fused_mlp = True
        config.fused_dropout_add_ln = True
        config.residual_in_fp32 = True
        config.rms_norm = True

        config.n_positions = 2048
        config.rotary_emb_fraction = 0.0
        config.rotary_emb_base = 0
        config.use_alibi = True
        # config.n_positions = 0
        # config.rotary_emb_fraction = 1.0
        # config.rotary_emb_base = 10000

        self.model = GPTModel(config, dtype=torch.float32)
        self.flash_attn_flag = True

    def make_init_hidden(self, max_seqlen, max_batch_size):
        return InferenceParams(max_seqlen, max_batch_size)

    def forward(self, hidden_states, attention_mask_in_length=None, inference_params=None):
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.bfloat16)
        with torch.cuda.amp.autocast():
            result = self.model.forward(hidden_states, attention_mask_in_length, inference_params)
        result = result.to(original_dtype)
        return result



