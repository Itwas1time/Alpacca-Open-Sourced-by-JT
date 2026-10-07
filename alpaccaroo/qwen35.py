# Alpaccaroo - Qwen3.5/Qwen3.8 hybrid architecture contract.
# MIT License. See LICENSE.
"""Allocation-free Qwen35 GGUF manifest inspection.

The product model may be named Qwen3.8, but compatible GGUF files identify
the runtime architecture as ``qwen35``.  This module treats that header value,
metadata and tensor table as authoritative and deliberately contains no
execution fallback for approximately similar model names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

from .gguf import (
    GGUFFile,
    T_ARRAY,
    T_BOOL,
    T_FLOAT32,
    T_INT32,
    T_STRING,
    T_UINT32,
)
from .weights import (
    ManifestError,
    metadata_inventory,
    stream_sha256,
    validate_tensor_layout,
)
from .memory import LayerMemorySpec


ARCHITECTURE = "qwen35"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Qwen35Config:
    block_count: int
    main_layer_count: int
    nextn_predict_layers: int
    embedding_length: int
    feed_forward_length: int
    vocabulary_size: int
    query_heads: int
    kv_heads: int
    key_length: int
    value_length: int
    context_length: int
    rms_epsilon: float
    rope_freq_base: float
    rope_dimension_count: int
    rope_dimension_sections: tuple[int, int, int, int]
    conv_kernel: int
    inner_size: int
    state_size: int
    time_step_rank: int
    group_count: int

    @property
    def conv_width(self) -> int:
        return self.inner_size + 2 * self.group_count * self.state_size


def _encoded_type(gguf: GGUFFile, key: str) -> tuple[int, int | None] | None:
    return gguf.metadata_types.get(key)


def _required_positive_uint32(gguf: GGUFFile, key: str) -> int:
    raw = gguf.metadata.get(key)
    encoded = _encoded_type(gguf, key)
    if raw is None:
        raise ManifestError(f"missing required metadata {key}")
    if encoded != (T_UINT32, None):
        actual = "missing type" if encoded is None else encoded[0]
        raise ManifestError(f"metadata {key} must use GGUF UINT32, got {actual}")
    value = int(raw)
    if value <= 0:
        raise ManifestError(f"metadata {key} must be positive, got {value}")
    return value


def _optional_nonnegative_uint32(
    gguf: GGUFFile, key: str, default: int = 0
) -> int:
    if key not in gguf.metadata:
        return default
    encoded = _encoded_type(gguf, key)
    if encoded != (T_UINT32, None):
        actual = "missing type" if encoded is None else encoded[0]
        raise ManifestError(f"metadata {key} must use GGUF UINT32, got {actual}")
    value = int(gguf.metadata[key])
    if value < 0:
        raise ManifestError(f"metadata {key} must be non-negative, got {value}")
    return value


def _required_positive_float32(gguf: GGUFFile, key: str) -> float:
    raw = gguf.metadata.get(key)
    encoded = _encoded_type(gguf, key)
    if raw is None:
        raise ManifestError(f"missing required metadata {key}")
    if encoded != (T_FLOAT32, None):
        actual = "missing type" if encoded is None else encoded[0]
        raise ManifestError(f"metadata {key} must use GGUF FLOAT32, got {actual}")
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ManifestError(f"metadata {key} must be positive and finite, got {value}")
    return value


def _required_int32_array(
    gguf: GGUFFile, key: str, length: int
) -> tuple[int, ...]:
    raw = gguf.metadata.get(key)
    encoded = _encoded_type(gguf, key)
    if raw is None:
        raise ManifestError(f"missing required metadata {key}")
    if encoded != (T_ARRAY, T_INT32):
        raise ManifestError(f"metadata {key} must use GGUF ARRAY[INT32]")
    if not isinstance(raw, list) or len(raw) != length:
        actual = len(raw) if isinstance(raw, list) else type(raw).__name__
        raise ManifestError(f"metadata {key} must contain {length} values, got {actual}")
    values = tuple(int(item) for item in raw)
    if any(item < 0 for item in values):
        raise ManifestError(f"metadata {key} cannot contain negative values: {values}")
    return values


def _required_string(gguf: GGUFFile, key: str) -> str:
    raw = gguf.metadata.get(key)
    if raw is None:
        raise ManifestError(f"missing required metadata {key}")
    if _encoded_type(gguf, key) != (T_STRING, None):
        raise ManifestError(f"metadata {key} must use GGUF STRING")
    value = str(raw)
    if not value:
        raise ManifestError(f"metadata {key} cannot be empty")
    return value


def _required_array(
    gguf: GGUFFile, key: str, element_type: int
) -> list[Any]:
    raw = gguf.metadata.get(key)
    if raw is None:
        raise ManifestError(f"missing required metadata {key}")
    if _encoded_type(gguf, key) != (T_ARRAY, element_type):
        expected = "STRING" if element_type == T_STRING else "INT32"
        raise ManifestError(f"metadata {key} must use GGUF ARRAY[{expected}]")
    if not isinstance(raw, list):
        raise ManifestError(f"metadata {key} did not decode as an array")
    return raw


def _validate_tokenizer(gguf: GGUFFile) -> int:
    model = _required_string(gguf, "tokenizer.ggml.model")
    pre = _required_string(gguf, "tokenizer.ggml.pre")
    if model != "gpt2":
        raise ManifestError(
            f"tokenizer.ggml.model must be 'gpt2' for Qwen35, got {model!r}"
        )
    if pre != "qwen35":
        raise ManifestError(
            f"tokenizer.ggml.pre must be 'qwen35', got {pre!r}"
        )

    tokens = _required_array(gguf, "tokenizer.ggml.tokens", T_STRING)
    if not tokens:
        raise ManifestError("metadata tokenizer.ggml.tokens cannot be empty")
    vocabulary_size = len(tokens)
    token_types = _required_array(gguf, "tokenizer.ggml.token_type", T_INT32)
    if len(token_types) != vocabulary_size:
        raise ManifestError(
            f"tokenizer.ggml.token_type contains {len(token_types)} entries, "
            f"expected {vocabulary_size}"
        )
    invalid_token_types = sorted({
        int(value) for value in token_types if int(value) < 1 or int(value) > 6
    })
    if invalid_token_types:
        raise ManifestError(
            f"tokenizer.ggml.token_type contains invalid values "
            f"{invalid_token_types}; expected GGML token types 1..6"
        )
    _required_array(gguf, "tokenizer.ggml.merges", T_STRING)
    _required_string(gguf, "tokenizer.chat_template")

    for key in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id"):
        if key not in gguf.metadata:
            raise ManifestError(f"missing required metadata {key}")
    for key in (
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.separator_token_id",
    ):
        if key not in gguf.metadata:
            continue
        token_id = _optional_nonnegative_uint32(gguf, key)
        if token_id >= vocabulary_size:
            raise ManifestError(
                f"metadata {key} is {token_id}, outside vocabulary size "
                f"{vocabulary_size}"
            )
    for key in ("tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"):
        if key in gguf.metadata and _encoded_type(gguf, key) != (T_BOOL, None):
            raise ManifestError(f"metadata {key} must use GGUF BOOL")
    return vocabulary_size


def _read_config(gguf: GGUFFile) -> Qwen35Config:
    architecture_type = _encoded_type(gguf, "general.architecture")
    if architecture_type is None or architecture_type[0] != T_STRING:
        raise ManifestError("metadata general.architecture must use GGUF STRING")
    if gguf.architecture != ARCHITECTURE:
        raise ManifestError(
            f"architecture mismatch: expected {ARCHITECTURE!r}, got {gguf.architecture!r}"
        )

    block_count = _required_positive_uint32(gguf, "qwen35.block_count")
    nextn = _optional_nonnegative_uint32(gguf, "qwen35.nextn_predict_layers", 0)
    if nextn >= block_count:
        raise ManifestError(
            f"qwen35.nextn_predict_layers ({nextn}) must be smaller than block_count ({block_count})"
        )
    if nextn > 1:
        raise ManifestError(
            "qwen35.nextn_predict_layers supports at most one disabled appended "
            f"MTP layer in this release, got {nextn}"
        )
    hidden = _required_positive_uint32(gguf, "qwen35.embedding_length")
    ffn = _required_positive_uint32(gguf, "qwen35.feed_forward_length")
    query_heads = _required_positive_uint32(gguf, "qwen35.attention.head_count")
    kv_heads = _required_positive_uint32(gguf, "qwen35.attention.head_count_kv")
    key_length = _required_positive_uint32(gguf, "qwen35.attention.key_length")
    value_length = _required_positive_uint32(gguf, "qwen35.attention.value_length")
    context = _required_positive_uint32(gguf, "qwen35.context_length")
    rms_epsilon = _required_positive_float32(
        gguf, "qwen35.attention.layer_norm_rms_epsilon"
    )
    rope_base = _required_positive_float32(gguf, "qwen35.rope.freq_base")
    rope_dimension_count = _required_positive_uint32(
        gguf, "qwen35.rope.dimension_count"
    )
    rope_sections = _required_int32_array(
        gguf, "qwen35.rope.dimension_sections", 4
    )
    conv_kernel = _required_positive_uint32(gguf, "qwen35.ssm.conv_kernel")
    inner_size = _required_positive_uint32(gguf, "qwen35.ssm.inner_size")
    state_size = _required_positive_uint32(gguf, "qwen35.ssm.state_size")
    time_step_rank = _required_positive_uint32(gguf, "qwen35.ssm.time_step_rank")
    group_count = _required_positive_uint32(gguf, "qwen35.ssm.group_count")

    if key_length != value_length:
        raise ManifestError(
            f"Qwen35 full attention requires equal key/value lengths, got "
            f"{key_length}/{value_length}"
        )
    if query_heads % kv_heads:
        raise ManifestError(
            f"attention.head_count {query_heads} must be divisible by "
            f"attention.head_count_kv {kv_heads}"
        )
    if inner_size % time_step_rank:
        raise ManifestError(
            f"ssm.inner_size {inner_size} is not divisible by ssm.time_step_rank {time_step_rank}"
        )
    recurrent_value_dim = inner_size // time_step_rank
    if recurrent_value_dim != state_size:
        raise ManifestError(
            f"recurrent value dimension {recurrent_value_dim} must equal "
            f"ssm.state_size {state_size}"
        )
    if time_step_rank % group_count:
        raise ManifestError(
            f"ssm.time_step_rank {time_step_rank} must be divisible by "
            f"ssm.group_count {group_count}"
        )
    if rope_dimension_count != sum(rope_sections) * 2:
        raise ManifestError(
            f"qwen35.rope.dimension_count {rope_dimension_count} must equal "
            f"2 * sum(dimension_sections) ({sum(rope_sections) * 2})"
        )
    if rope_dimension_count > key_length:
        raise ManifestError(
            f"qwen35.rope.dimension_sections {rope_sections} rotate more than "
            f"attention.key_length {key_length}"
        )

    vocabulary_size = _validate_tokenizer(gguf)
    metadata_vocab = gguf.metadata.get("qwen35.vocab_size")
    if metadata_vocab is not None:
        declared = _required_positive_uint32(gguf, "qwen35.vocab_size")
        if declared != vocabulary_size:
            raise ManifestError(
                f"qwen35.vocab_size is {declared}, tokenizer has {vocabulary_size} entries"
            )

    return Qwen35Config(
        block_count=block_count,
        main_layer_count=block_count - nextn,
        nextn_predict_layers=nextn,
        embedding_length=hidden,
        feed_forward_length=ffn,
        vocabulary_size=vocabulary_size,
        query_heads=query_heads,
        kv_heads=kv_heads,
        key_length=key_length,
        value_length=value_length,
        context_length=context,
        rms_epsilon=rms_epsilon,
        rope_freq_base=rope_base,
        rope_dimension_count=rope_dimension_count,
        rope_dimension_sections=rope_sections,
        conv_kernel=conv_kernel,
        inner_size=inner_size,
        state_size=state_size,
        time_step_rank=time_step_rank,
        group_count=group_count,
    )


def _layer_schedule(gguf: GGUFFile, config: Qwen35Config) -> tuple[list[str], str]:
    key = "qwen35.attention.recurrent_layers"
    raw = gguf.metadata.get(key)
    encoded = _encoded_type(gguf, key)
    if raw is not None:
        if encoded is None or encoded[0] != T_ARRAY:
            raise ManifestError(f"metadata {key} must be a GGUF array")
        if encoded[1] == T_BOOL:
            if len(raw) != config.block_count:
                raise ManifestError(
                    f"metadata {key} contains {len(raw)} flags, expected {config.block_count}"
                )
            recurrent = {index for index, flag in enumerate(raw) if bool(flag)}
            source = "qwen35.attention.recurrent_layers boolean flags"
        elif encoded[1] == T_UINT32:
            if len(raw) != config.block_count:
                raise ManifestError(
                    f"metadata {key} contains {len(raw)} flags, expected {config.block_count}"
                )
            if any(int(flag) not in (0, 1) for flag in raw):
                raise ManifestError(f"metadata {key} integer flags must be zero or one")
            recurrent = {index for index, flag in enumerate(raw) if int(flag)}
            source = "qwen35.attention.recurrent_layers integer flags"
        else:
            raise ManifestError(
                f"metadata {key} must contain BOOL or UINT32 values"
            )
        invalid = sorted(index for index in recurrent if index < 0 or index >= config.main_layer_count)
        if invalid:
            raise ManifestError(
                f"metadata {key} identifies non-main layer(s) as recurrent: {invalid}"
            )
    else:
        interval_key = "qwen35.full_attention_interval"
        interval = _optional_nonnegative_uint32(gguf, interval_key, 4)
        if interval <= 0:
            raise ManifestError(f"metadata {interval_key} must be positive, got {interval}")
        recurrent = {
            index for index in range(config.main_layer_count)
            if (index + 1) % interval != 0
        }
        source = (
            interval_key if interval_key in gguf.metadata
            else "upstream-compatible default full_attention_interval=4"
        )

    schedule: list[str] = []
    for index in range(config.block_count):
        if index >= config.main_layer_count:
            schedule.append("mtp-disabled")
        else:
            schedule.append("recurrent" if index in recurrent else "attention")
    if "attention" not in schedule[:config.main_layer_count]:
        raise ManifestError("Qwen35 main stack contains no full-attention layer")
    if "recurrent" not in schedule[:config.main_layer_count]:
        raise ManifestError("Qwen35 main stack contains no recurrent layer")
    return schedule, source


def qwen35_memory_specs(
    config: Qwen35Config, schedule: list[str]
) -> tuple[LayerMemorySpec, ...]:
    """Derive immutable heterogeneous-state descriptors from the manifest."""
    if len(schedule) != config.block_count:
        raise ManifestError(
            f"layer schedule has {len(schedule)} entries, expected "
            f"{config.block_count}"
        )
    specs: list[LayerMemorySpec] = []
    for index, kind in enumerate(schedule):
        if kind == "attention":
            specs.append(LayerMemorySpec(
                layer_index=index,
                kind="attention",
                kv_heads=config.kv_heads,
                key_dim=config.key_length,
                value_dim=config.value_length,
            ))
        elif kind == "recurrent":
            specs.append(LayerMemorySpec(
                layer_index=index,
                kind="recurrent",
                conv_channels=config.conv_width,
                conv_kernel=config.conv_kernel,
                recurrent_key_heads=config.group_count,
                recurrent_value_heads=config.time_step_rank,
                recurrent_key_dim=config.state_size,
                recurrent_value_dim=config.state_size,
            ))
        elif kind == "mtp-disabled":
            specs.append(LayerMemorySpec(layer_index=index, kind="none"))
        else:
            raise ManifestError(
                f"layer {index} has unsupported schedule kind {kind!r}"
            )
    return tuple(specs)


def _expected_tensors(config: Qwen35Config, schedule: list[str]) -> tuple[dict[str, tuple[int, ...]], set[str]]:
    hidden = config.embedding_length
    ffn = config.feed_forward_length
    vocab = config.vocabulary_size
    expected: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (hidden, vocab),
        "output_norm.weight": (hidden,),
    }
    optional = {"output.weight"}
    if "output.weight" in optional:
        # Kept separate so absence has the precise tied-embedding meaning.
        pass

    for index, kind in enumerate(schedule):
        prefix = f"blk.{index}."
        expected.update({
            prefix + "attn_norm.weight": (hidden,),
            prefix + "post_attention_norm.weight": (hidden,),
            prefix + "ffn_gate.weight": (hidden, ffn),
            prefix + "ffn_up.weight": (hidden, ffn),
            prefix + "ffn_down.weight": (ffn, hidden),
        })
        if kind in ("attention", "mtp-disabled"):
            expected.update({
                prefix + "attn_q.weight": (
                    hidden, 2 * config.query_heads * config.key_length
                ),
                prefix + "attn_k.weight": (
                    hidden, config.kv_heads * config.key_length
                ),
                prefix + "attn_v.weight": (
                    hidden, config.kv_heads * config.value_length
                ),
                prefix + "attn_output.weight": (
                    config.query_heads * config.value_length, hidden
                ),
                prefix + "attn_q_norm.weight": (config.key_length,),
                prefix + "attn_k_norm.weight": (config.key_length,),
            })
        else:
            expected.update({
                prefix + "attn_qkv.weight": (hidden, config.conv_width),
                prefix + "attn_gate.weight": (hidden, config.inner_size),
                prefix + "ssm_conv1d.weight": (config.conv_kernel, config.conv_width),
                prefix + "ssm_dt.bias": (config.time_step_rank,),
                prefix + "ssm_a": (config.time_step_rank,),
                prefix + "ssm_alpha.weight": (hidden, config.time_step_rank),
                prefix + "ssm_beta.weight": (hidden, config.time_step_rank),
                prefix + "ssm_norm.weight": (config.state_size,),
                prefix + "ssm_out.weight": (config.inner_size, hidden),
            })

        if kind == "mtp-disabled":
            expected.update({
                prefix + "nextn.eh_proj.weight": (2 * hidden, hidden),
                prefix + "nextn.enorm.weight": (hidden,),
                prefix + "nextn.hnorm.weight": (hidden,),
            })
            optional.update({
                prefix + "nextn.embed_tokens.weight",
                prefix + "nextn.shared_head_head.weight",
                prefix + "nextn.shared_head_norm.weight",
            })

    return expected, optional


def _validate_tensor_contract(
    gguf: GGUFFile,
    config: Qwen35Config,
    schedule: list[str],
    inventory: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    expected, optional = _expected_tensors(config, schedule)
    optional_shapes = {
        "output.weight": (config.embedding_length, config.vocabulary_size),
    }
    for index, kind in enumerate(schedule):
        if kind == "mtp-disabled":
            prefix = f"blk.{index}."
            optional_shapes.update({
                prefix + "nextn.embed_tokens.weight": (
                    config.embedding_length, config.vocabulary_size
                ),
                prefix + "nextn.shared_head_head.weight": (
                    config.embedding_length, config.vocabulary_size
                ),
                prefix + "nextn.shared_head_norm.weight": (config.embedding_length,),
            })

    missing = sorted(name for name in expected if name not in gguf.tensors)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ManifestError(f"missing required Qwen35 tensor(s): {preview}{suffix}")

    role_by_name: dict[str, str] = {}
    for name, shape in {**expected, **optional_shapes}.items():
        info = gguf.tensors.get(name)
        if info is None:
            continue
        if tuple(info.shape) != shape:
            raise ManifestError(
                f"tensor {name} has shape {info.shape}, expected {shape}"
            )
        if name.startswith("blk."):
            role_by_name[name] = name.split(".", 2)[2]
        else:
            role_by_name[name] = name

    classified = set(expected) | {name for name in optional if name in gguf.tensors}
    unclassified = sorted(set(gguf.tensors) - classified)
    unsupported_auxiliary = sorted(
        name for name in unclassified
        if name.startswith(("v.", "vision.", "mm.", "mmproj."))
        or ".vision" in name or ".mm_proj" in name
    )
    truly_unknown = sorted(set(unclassified) - set(unsupported_auxiliary))
    if truly_unknown:
        preview = ", ".join(truly_unknown[:8])
        suffix = " ..." if len(truly_unknown) > 8 else ""
        raise ManifestError(f"unclassified Qwen35 tensor(s): {preview}{suffix}")

    for record in inventory:
        name = str(record["name"])
        record["role"] = role_by_name.get(name, "unsupported-auxiliary")
        if name.startswith("blk."):
            try:
                record["layer_index"] = int(name.split(".", 2)[1])
            except (ValueError, IndexError):
                record["layer_index"] = None
        else:
            record["layer_index"] = None
    return unclassified, unsupported_auxiliary


def inspect_qwen35(
    path: str | Path,
    *,
    context: int = 0,
    sequences: int = 1,
    kv_dtype: str = "f32",
    cuda_vram_mb: float | None = None,
    cuda_reserve_mb: float = 1536.0,
    source_url: str | None = None,
    revision: str | None = None,
    calculate_hash: bool = True,
    vision_requested: bool = False,
    mtp_requested: bool = False,
) -> dict[str, Any]:
    """Inspect a Qwen35 GGUF without allocating or decoding model tensors."""
    if sequences <= 0:
        raise ManifestError(f"sequences must be positive, got {sequences}")
    if context < 0:
        raise ManifestError(f"context must be non-negative, got {context}")
    if kv_dtype.lower() not in ("f32", "f16"):
        raise ManifestError(f"kv_dtype must be f32 or f16, got {kv_dtype!r}")
    if not math.isfinite(cuda_reserve_mb) or cuda_reserve_mb < 0:
        raise ManifestError("cuda_reserve_mb must be finite and non-negative")
    if cuda_vram_mb is not None and (
        not math.isfinite(cuda_vram_mb) or cuda_vram_mb < 0
    ):
        raise ManifestError("cuda_vram_mb must be finite and non-negative")

    path = Path(path).expanduser().resolve()
    try:
        with GGUFFile.open(path, prefetch=False) as gguf:
            inventory, dtype_census = validate_tensor_layout(gguf)
            config = _read_config(gguf)
            schedule, schedule_source = _layer_schedule(gguf, config)
            memory_specs = qwen35_memory_specs(config, schedule)
            unused, unsupported_auxiliary = _validate_tensor_contract(
                gguf, config, schedule, inventory
            )
            if vision_requested:
                raise ManifestError(
                    "vision input was requested, but the Qwen35 text runtime "
                    "does not execute vision/multimodal tensors"
                )
            if mtp_requested:
                raise ManifestError(
                    "MTP/NextN execution was requested, but the Qwen35 text "
                    "runtime only inventories appended MTP tensors; use text "
                    "generation with MTP disabled"
                )
            metadata = metadata_inventory(gguf)
            file_size = gguf.file_size
            gguf_version = gguf.version
            alignment = gguf.alignment
            data_start = gguf.data_start
    except ManifestError:
        raise
    except (OSError, ValueError) as exc:
        raise ManifestError(str(exc)) from exc

    full_layers = sum(kind == "attention" for kind in schedule)
    recurrent_layers = sum(kind == "recurrent" for kind in schedule)
    requested_context = context or config.context_length
    if requested_context > config.context_length:
        raise ManifestError(
            f"requested context {requested_context} exceeds trained context "
            f"{config.context_length}"
        )
    kv_element_bytes = 4 if kv_dtype.lower() == "f32" else 2
    recurrent_matrix_bytes = (
        recurrent_layers
        * config.time_step_rank
        * config.state_size
        * config.state_size
        * 4
        * sequences
    )
    convolution_history_bytes = (
        recurrent_layers
        * (config.conv_kernel - 1)
        * config.conv_width
        * 4
        * sequences
    )
    kv_bytes = (
        full_layers
        * requested_context
        * config.kv_heads
        * (config.key_length + config.value_length)
        * kv_element_bytes
        * sequences
    )
    packed_weight_bytes = sum(
        int(item["packed_bytes"]) for item in inventory
        if item["role"] != "unsupported-auxiliary"
    )
    # This is a deliberately visible reserve, not a claim that the remaining
    # runtime workspace has already been kernel-qualified.
    reserve_bytes = int(cuda_reserve_mb * 1024 * 1024)
    logits_and_activation_bytes = (
        config.vocabulary_size * 4
        + max(config.embedding_length, config.inner_size, config.conv_width) * 4
    ) * sequences
    planned_cuda_bytes = (
        packed_weight_bytes + recurrent_matrix_bytes + convolution_history_bytes
        + kv_bytes + logits_and_activation_bytes + reserve_bytes
    )
    vram_budget_bytes = (
        None if cuda_vram_mb is None else int(cuda_vram_mb * 1024 * 1024)
    )

    layers = [
        {"index": index, "kind": kind, "main_text_stack": index < config.main_layer_count}
        for index, kind in enumerate(schedule)
    ]
    tokenizer_keys = sorted(
        key for key in metadata if key.startswith("tokenizer.")
    )
    warnings = []
    if unsupported_auxiliary:
        warnings.append("vision/multimodal auxiliary tensors are present but unsupported")
    if config.nextn_predict_layers:
        warnings.append(
            f"{config.nextn_predict_layers} MTP/NextN layer(s) are inventoried but disabled"
        )
    if sequences > 1:
        warnings.append(
            "multi-sequence batching is requested for planning but execution is unsupported"
        )
    if vram_budget_bytes is not None and planned_cuda_bytes > vram_budget_bytes:
        warnings.append(
            f"conservative CUDA plan exceeds configured budget by "
            f"{planned_cuda_bytes - vram_budget_bytes} bytes"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "path": str(path),
            "url": source_url,
            "revision": revision,
            "filename": path.name,
            "byte_size": file_size,
            "sha256": stream_sha256(path) if calculate_hash else None,
        },
        "gguf": {
            "version": gguf_version,
            "architecture": ARCHITECTURE,
            "alignment": alignment,
            "data_start": data_start,
            "metadata": metadata,
            "tokenizer_metadata_keys": tokenizer_keys,
        },
        "model": {
            **asdict(config),
            "conv_width": config.conv_width,
            "recurrent_layers": recurrent_layers,
            "full_attention_layers": full_layers,
        },
        "layer_schedule_source": schedule_source,
        "layers": layers,
        "layer_memory_specs": [spec.descriptor() for spec in memory_specs],
        "tensors": inventory,
        "dtype_census": dtype_census,
        "unused_tensors": unused,
        "unsupported_features": {
            "vision": {
                "supported": False,
                "present": bool(unsupported_auxiliary),
                "requested": vision_requested,
            },
            "mtp_execution": {
                "supported": False,
                "present": bool(config.nextn_predict_layers),
                "requested": mtp_requested,
            },
            "speculative_decoding": {
                "supported": False,
                "present": False,
                "requested": False,
            },
            "multi_sequence_batching": {
                "supported": False,
                "present": False,
                "requested": sequences > 1,
            },
        },
        "memory_plan": {
            "scope": "allocation estimate only; execution kernels are unqualified",
            "placement_assumption": "all classified weights are device-resident",
            "required_weight_dtypes": sorted(dtype_census),
            "dtype_kernel_qualification": "not-tested",
            "activation_workspace_assumption": (
                "one logits vector plus one maximum-width activation, multiplied "
                "by sequences, plus the configured opaque reserve"
            ),
            "requested_context": requested_context,
            "sequences": sequences,
            "kv_dtype": kv_dtype.lower(),
            "packed_weight_bytes": packed_weight_bytes,
            "recurrent_matrix_bytes": recurrent_matrix_bytes,
            "convolution_history_bytes": convolution_history_bytes,
            "attention_kv_bytes": kv_bytes,
            "logits_and_activation_bytes": logits_and_activation_bytes,
            "cuda_reserve_bytes": reserve_bytes,
            "planned_cuda_bytes": planned_cuda_bytes,
            "configured_vram_budget_bytes": vram_budget_bytes,
            "allocation_fits_configured_vram_budget": (
                None if vram_budget_bytes is None
                else planned_cuda_bytes <= vram_budget_bytes
            ),
            "execution_feasible": False,
        },
        "warnings": warnings,
    }
