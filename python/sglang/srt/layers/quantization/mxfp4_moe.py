import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from sgl_kernel import sgl_per_tensor_quant_fp8
import triton
import triton.language as tl

import triton_kernels
import triton_kernels.swiglu
from triton_kernels.matmul_ogs import (FlexCtx, FnSpecs, FusedActivation,
                                       PrecisionConfig, matmul_ogs, InFlexData)
from triton_kernels.routing import routing
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details import layout
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp_torch


def shuffle_for_activation_kernel(weight: torch.Tensor) -> torch.Tensor:
    temp_weight = weight.clone()
    last_dim = weight.shape[-1]
    if weight.dim() == 3:
        weight[:, :, 1::2] = temp_weight[:, :, last_dim // 2:]
        weight[:, :, 0::2] = temp_weight[:, :, 0:last_dim // 2]
    elif weight.dim() == 2:
        weight[:, 1::2] = temp_weight[:, last_dim // 2:]
        weight[:, 0::2] = temp_weight[:, 0:last_dim // 2]
    return weight

def quantize_to_mxfp4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = tensor.transpose(1, 2).contiguous()
    tensor_fp4, tensor_scales = downcast_to_mxfp_torch(tensor, torch.uint8, axis=1)
    tensor_fp4 = tensor_fp4.transpose(1, 2).contiguous()
    tensor_scales = tensor_scales.transpose(1, 2).contiguous()
    return tensor_fp4, tensor_scales

@triton.jit
def set_to_zero_kernel(output_ptr):
    """
    Triton kernel to set a single-element tensor to 0.0.
    """
    tl.store(output_ptr, 0.0)

def quantize_fp8_per_tensor(
                input_q: torch.Tensor,
                scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_type_ = torch.float8_e4m3fn
    output_q = torch.empty_like(input_q, dtype=fp8_type_, device=input_q.device)
    is_static = True
    if scale is None:
        scale = torch.empty(1, dtype=torch.float32, device=input_q.device)
        set_to_zero_kernel[(1,)](scale,)
        is_static = False
    sgl_per_tensor_quant_fp8(input_q, output_q, scale, is_static)
    return output_q, scale

def swizzle_weight_and_scale(w: torch.Tensor, w_scale: torch.Tensor):
    w = w.transpose(-1, -2).contiguous().transpose(-1, -2)
    num_warps = int(os.getenv("TRITON_MOE_MXFP4_NUM_WARPS", 4))
    assert num_warps in [4, 8], \
        f"TRITON_MOE_MXFP4_NUM_WARPS should be 4 or 8, got {num_warps}"
    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
        mx_axis=1)
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=1, num_warps=num_warps)
    opt = {"value_layout": value_layout, "value_layout_opts": value_layout_opts, \
            "scale_layout": scale_layout, "scale_layout_opts": scale_layout_opts}

    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), opt["value_layout"],
                       **opt["value_layout_opts"])
    w_scale = convert_layout(wrap_torch_tensor(w_scale), opt["scale_layout"],
                             **opt["scale_layout_opts"])
    return w, w_scale

def maybe_remove_padding(gemm_output: torch.Tensor,
                         expected_size: int) -> torch.Tensor:
    assert gemm_output.dim() == 2
    if gemm_output.shape[-1] != expected_size:
        assert gemm_output.shape[
            -1] % 256 == 0, "The padding is not done correctly"
        gemm_output = gemm_output[:, :expected_size]
    return gemm_output

def swiglu_torch(a: torch.Tensor, alpha: float, beta: float,
                 limit: Optional[float]) -> torch.Tensor:
    a_glu = a[..., ::2]
    if limit is not None:
        a_glu = a_glu.clamp(max=limit)
    a_linear = a[..., 1::2]
    if limit is not None:
        a_linear = a_linear.clamp(min=-limit, max=limit)

    out_glu = a_glu * torch.sigmoid(alpha * a_glu)
    out = out_glu * (a_linear + beta)
    return out

def fused_experts_mxfp4_oai(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    expert_logits: torch.Tensor, # (num_tokens, num_experts)
    top_k: int,
    fc31_input_dequant: torch.Tensor,
    fc2_input_dequant: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    activation: str,  # "swiglu"
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    dtype: torch.dtype = torch.bfloat16,
    activation_dtype: torch.dtype = torch.float8_e4m3fn,
    intermediate_size: int = 0,
    hidden_size: int = 0,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor:
    if activation_dtype == torch.float8_e4m3fn:
        hidden_states, hidden_states_scale = quantize_fp8_per_tensor(hidden_states, fc31_input_dequant)
    else:
        hidden_states = hidden_states
    gemm1_weights = w13
    gemm1_scales = w13_scale
    gemm2_weights = w2
    gemm2_scales = w2_scale
    top_k = top_k

    num_experts = expert_logits.shape[1]
    if num_experts > 1:
        rdata, gather_indx, scatter_indx = routing(expert_logits, top_k)
    else:
        rdata, gather_indx, scatter_indx = None, None, None

    if activation_dtype == torch.float8_e4m3fn:
        flex_ctx_1 = FlexCtx(
            lhs_data=InFlexData(scale=hidden_states_scale), )
    else:
        flex_ctx_1 = FlexCtx()
    pc1 = PrecisionConfig(weight_scale=gemm1_scales,
                          flex_ctx=flex_ctx_1,
                          allow_tf32=False,
                          out_dtype=dtype)
    alpha = swiglu_alpha or 1.0
    beta = swiglu_beta or 0.0
    if beta == 1.0:
        act = FusedActivation(
            FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn,
                    ("alpha", "limit")), (alpha, swiglu_limit), 2)

        act_out = matmul_ogs(hidden_states,
                            gemm1_weights,
                            w1_bias,
                            rdata,
                            gather_indx=gather_indx,
                            precision_config=pc1,
                            fused_activation=act)
    else:
        act_out = matmul_ogs(hidden_states,
                            gemm1_weights,
                            w1_bias,
                            rdata,
                            gather_indx=gather_indx,
                            precision_config=pc1)
        act_out = swiglu_torch(act_out, alpha, beta, swiglu_limit)


    act_out = maybe_remove_padding(
            act_out, intermediate_size).contiguous()

    if activation_dtype == torch.float8_e4m3fn:
        act_out, act_scale = quantize_fp8_per_tensor(act_out, fc2_input_dequant)

    if activation_dtype == torch.float8_e4m3fn:
        flex_ctx_2 = FlexCtx(lhs_data=InFlexData(scale=act_scale), )
    else:
        flex_ctx_2 = FlexCtx()
    pc2 = PrecisionConfig(weight_scale=gemm2_scales,
                          flex_ctx=flex_ctx_2,
                          allow_tf32=False,
                          out_dtype=dtype)

    gemm2_output = matmul_ogs(
        act_out,
        gemm2_weights,
        w2_bias,  # Bias
        rdata,
        scatter_indx=scatter_indx,
        precision_config=pc2,
        gammas=rdata.gate_scal if rdata else None)
    gemm2_output = maybe_remove_padding(gemm2_output, hidden_size).contiguous()
    return gemm2_output
