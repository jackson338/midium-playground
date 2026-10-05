"""One local completion from mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp.

The checkpoint is qwen4_exp. mlx-vlm 0.7.0 loads that architecture, and its
n-gram table reader only dequantizes 4-bit rows. This pack stores that table
at 6-bit, group size 32, so the reader is patched in-process before load.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
from pathlib import Path

from playground.config import ROOT

HF_REPO = "mlx-community/Qwen3.8-Flash-Next-oQ6e-mtp"
MODEL_DIR = ROOT / "data" / "models" / "qwen3.8-flash-next-oq6e-mtp"
_SIX_BIT_LAYOUT = (("weight", "U32"), ("scales", "BF16"), ("biases", "BF16"))


def unpacked_row_width(packed_cols: int, bits: int) -> int:
    """Unpacked embedding width. MLX stores `bits` per value inside uint32 words."""
    if bits <= 0 or packed_cols <= 0 or (packed_cols * 32) % bits:
        raise ValueError(f"packed width {packed_cols} is not a {bits}-bit row")
    return packed_cols * 32 // bits


def weights_ready(path: Path) -> bool:
    return (path / "config.json").is_file() and (path / "model.safetensors.index.json").is_file()


def ple_dir() -> Path:
    return MODEL_DIR.parent / f"{MODEL_DIR.name}-ple"


def ple_ready(path: Path) -> bool:
    return (
        (path / "ple-store.json").is_file()
        and (path / "config.json").is_file()
        and (path / "model.safetensors.index.json").is_file()
    )


def snapshot_download(repo: str, local_dir: str) -> None:
    from huggingface_hub import snapshot_download as download

    download(repo, local_dir=local_dir)


def ensure_weights() -> Path:
    if weights_ready(MODEL_DIR):
        return MODEL_DIR
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_REPO}", file=sys.stderr, flush=True)
    snapshot_download(HF_REPO, str(MODEL_DIR))
    if not weights_ready(MODEL_DIR):
        raise SystemExit(f"Download of {HF_REPO} did not produce config.json and the weight index.")
    return MODEL_DIR


def install_six_bit_ple(ple) -> None:
    """Teach mlx-vlm's PLE reader to index and dequantize 6-bit group-32 rows."""
    if getattr(ple, "_playground_six_bit", False):
        return
    ple._QUANTIZATION_LAYOUTS[(6, 32, "affine")] = _SIX_BIT_LAYOUT

    def build_quantized_ple_manifest(model_path, output_path, *, cache_rows=0):
        model_path = Path(model_path).resolve()
        output_path = Path(output_path).resolve()
        index = json.loads((model_path / "model.safetensors.index.json").read_text())
        config = json.loads((model_path / "config.json").read_text())
        weight_map = index["weight_map"]
        prefixes = {key.rsplit(".", 1)[0] for key in weight_map if ple.PLE_MARKER in key}
        if not prefixes:
            raise ValueError("checkpoint contains no Qwen4-Exp PLE shards")

        headers = {}

        def tensor_descriptor(key):
            file_name = weight_map[key]
            if file_name not in headers:
                with (model_path / file_name).open("rb") as stream:
                    header_size = struct.unpack("<Q", stream.read(8))[0]
                    headers[file_name] = (
                        8 + header_size,
                        json.loads(stream.read(header_size)),
                    )
            payload_start, header = headers[file_name]
            info = header[key]
            return {
                "file": file_name,
                "offset": payload_start + int(info["data_offsets"][0]),
                "dtype": info["dtype"],
                "shape": info["shape"],
            }

        shards = []
        row_start = 0
        manifest_quantization = None
        tensor_layout = None
        row_width = None
        ordered = sorted(prefixes, key=lambda value: int(value.rsplit(".", 1)[1]))
        for prefix in ordered:
            quantization = config.get("quantization_config") or config.get("quantization", {})
            params = quantization.get(prefix, quantization)
            current = {key: params.get(key) for key in ("bits", "group_size", "mode")}
            current_layout = ple._quantization_layout(current)
            if manifest_quantization is None:
                manifest_quantization = current
                tensor_layout = current_layout
            elif current != manifest_quantization:
                raise ValueError("all PLE shards must use the same quantization")
            tensors = {
                name: tensor_descriptor(f"{prefix}.{name}") for name, _dtype in tensor_layout
            }
            if any(tensors[name]["dtype"] != dtype for name, dtype in tensor_layout):
                raise ValueError(f"PLE tensor {prefix!r} has an invalid dtype layout")
            weight = tensors["weight"]
            if len(weight["shape"]) != 2:
                raise ValueError(f"PLE tensor {prefix!r} has an invalid shape layout")
            packed_cols = int(weight["shape"][1])
            width = unpacked_row_width(packed_cols, int(current["bits"]))
            if row_width is None:
                row_width = width
            elif width != row_width:
                raise ValueError("PLE shards do not share a row width")
            row_count = int(weight["shape"][0])
            groups = width // int(current["group_size"])
            if any(
                tensors[name]["shape"] != [row_count, groups]
                for name, _dtype in tensor_layout
                if name != "weight"
            ):
                raise ValueError(f"PLE tensor {prefix!r} has an invalid shape layout")
            shards.append({"row_start": row_start, "row_count": row_count, **tensors})
            row_start += row_count
        manifest = {
            "version": ple.QuantizedMMapNGramEmbedding.MANIFEST_VERSION,
            "source_root": os.path.relpath(model_path, output_path.parent),
            "layout": "safetensors_ranges",
            "row_width": row_width,
            "row_count": row_start,
            "quantization": manifest_quantization,
            "cache_rows": int(cache_rows),
            "shards": shards,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    def _init_safetensors_ranges(self, manifest, source_root):
        self._shards = []
        expected_start = 0
        bits = int(self.quantization["bits"])
        for entry in manifest["shards"]:
            start = int(entry["row_start"])
            count = int(entry["row_count"])
            if start != expected_start or count <= 0:
                raise ValueError("PLE shards must be positive, contiguous, and ordered")
            arrays = {}
            for name, expected_dtype in self._tensor_layout:
                tensor = entry[name]
                if tensor["dtype"] != expected_dtype:
                    raise ValueError(f"unexpected {name} dtype: {tensor['dtype']}")
                file_name = Path(tensor["file"])
                if file_name.is_absolute() or file_name.name != str(file_name):
                    raise ValueError("tensor files must be filenames under source_root")
                path = source_root / file_name
                offset = int(tensor["offset"])
                shape = tuple(int(dim) for dim in tensor["shape"])
                if shape[0] != count:
                    raise ValueError(f"{name} row count does not match shard")
                dtype = ple._DTYPES[expected_dtype]
                byte_count = int(ple.np.prod(shape)) * dtype.itemsize
                if offset < 0 or path.stat().st_size < offset + byte_count:
                    raise ValueError(f"{name} byte range exceeds {path}")
                arrays[name] = ple.np.memmap(
                    path,
                    dtype=dtype,
                    mode="r",
                    offset=offset,
                    shape=shape,
                )
            packed_cols = arrays["weight"].shape[1]
            if packed_cols * 32 != self.row_width * bits:
                raise ValueError("packed weight width does not match row_width")
            groups = self.row_width // self._group_size
            expected_aux_shape = (groups,)
            if any(
                arrays[name].shape[1:] != expected_aux_shape
                for name in self._tensor_names
                if name != "weight"
            ):
                raise ValueError("auxiliary tensor shape does not match group layout")
            self._shards.append((start, start + count, arrays))
            expected_start += count
        self.row_count = expected_start

    def _lookup(self, row_ids):
        started = ple.time.perf_counter()
        ids = ple.np.asarray(row_ids).astype(ple.np.int64, copy=False)
        if ids.size and (ids.min() < 0 or ids.max() >= self.row_count):
            raise IndexError("n-gram row outside mmap table")
        flat_ids = ids.reshape(-1)
        if not flat_ids.size:
            return ple.mx.zeros((*ids.shape, self.row_width), dtype=ple.mx.bfloat16)

        unique_ids, inverse = ple.np.unique(flat_ids, return_inverse=True)
        row_tensors = self._read_rows(unique_ids)
        duplicate_count = flat_ids.size - unique_ids.size
        self._hits += duplicate_count
        packed = ple.mx.array(row_tensors[0], dtype=ple.mx.uint32)
        bits = int(self.quantization["bits"])
        group_size = int(self.quantization["group_size"])
        if self.quantization["mode"] == "nvfp4":
            scales = ple.mx.array(row_tensors[1], dtype=ple.mx.uint8)
            values = ple.mx.dequantize(
                packed, scales, group_size=group_size, bits=bits, mode="nvfp4"
            )
        else:
            scales = ple.mx.array(ple._bfloat16_to_float32(row_tensors[1])).astype(ple.mx.bfloat16)
            biases = ple.mx.array(ple._bfloat16_to_float32(row_tensors[2])).astype(ple.mx.bfloat16)
            values = ple.mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)
        values = values[ple.mx.array(inverse, dtype=ple.mx.int64)]
        self._lookups += 1
        self._rows += ids.size
        self._elapsed += ple.time.perf_counter() - started
        return values.reshape(*ids.shape, self.row_width)

    ple.build_quantized_ple_manifest = build_quantized_ple_manifest
    ple.QuantizedMMapNGramEmbedding._init_safetensors_ranges = _init_safetensors_ranges
    ple.QuantizedMMapNGramEmbedding.__call__ = _lookup
    ple.QuantizedMMapNGramEmbedding.lookup = _lookup
    ple._playground_six_bit = True


def ensure_ple_view(source: Path) -> Path:
    from mlx_vlm.models.qwen4_exp import ple_storage

    install_six_bit_ple(ple_storage)
    view = ple_dir()
    if ple_ready(view):
        return view
    if view.exists():
        shutil.rmtree(view)
    ple_storage.prepare_external_ple_model(str(source), str(view))
    return view


def negotiate_metal_memory() -> None:
    """Match UCE's worker limits: working set, cache, and Apple's wired cap."""
    try:
        import mlx.core as mx
    except ImportError:
        return
    try:
        if not mx.metal.is_available():
            return
        info = mx.device_info()
        total = int(info.get("memory_size", 0))
        apple_max = int(info.get("max_recommended_working_set_size", 0))
    except Exception:
        return
    if total <= 0 or apple_max <= 0:
        return
    try:
        pct = float(os.getenv("COURIER_MEM_LIMIT_PERCENT", "0.84"))
    except (TypeError, ValueError):
        pct = 0.84
    try:
        cache_ratio = float(os.getenv("COURIER_MEM_CACHE_RATIO", "0.20"))
    except (TypeError, ValueError):
        cache_ratio = 0.20
    cache_ratio = max(0.0, min(0.5, cache_ratio))
    raw = total * pct
    bypass = os.getenv("COURIER_ALLOW_EXCEED_APPLE_MAX", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    working_set = int(raw if bypass or apple_max <= 0 else min(raw, apple_max))
    cache = int(working_set * cache_ratio)
    for name, value in (
        ("set_memory_limit", working_set),
        ("set_cache_limit", cache),
        ("set_wired_limit", apple_max),
    ):
        fn = getattr(mx, name, None)
        if fn is None:
            continue
        try:
            fn(value)
        except Exception:
            pass


def load_lenient(model_path: str):
    """mlx_vlm.load declares strict and does not forward it. Force it for one call."""
    import mlx.nn as nn
    from mlx_vlm import load

    original = nn.Module.load_weights

    def lenient(self, weights, strict=True):
        return original(self, weights, strict=False)

    nn.Module.load_weights = lenient
    try:
        return load(model_path)
    finally:
        nn.Module.load_weights = original


def render_prompt(processor, prompt: str, model_dir: Path) -> str:
    from mlx_vlm import apply_chat_template

    config = json.loads((model_dir / "config.json").read_text())
    text = apply_chat_template(
        processor,
        {"model_type": config["model_type"]},
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(text, str):
        raise SystemExit("Chat template did not return text.")
    return text


def complete(model, processor, prompt: str, max_tokens: int, model_dir: Path) -> str:
    from mlx_vlm.generate import stream_generate

    rendered = render_prompt(processor, prompt, model_dir)
    parts: list[str] = []
    for chunk in stream_generate(
        model,
        processor,
        prompt=rendered,
        image=None,
        max_tokens=max_tokens,
        temperature=0.0,
        enable_thinking=False,
        verbose=False,
    ):
        if chunk.text:
            parts.append(chunk.text)
    return "".join(parts)


def run_qwen_cli(prompt: str, max_tokens: int = 128) -> None:
    if max_tokens < 1:
        raise SystemExit("--max-tokens must be at least 1.")
    source = ensure_weights()
    view = ensure_ple_view(source)
    negotiate_metal_memory()
    print("Loading Qwen3.8 Flash Next", file=sys.stderr, flush=True)
    model, processor = load_lenient(str(view))
    print(complete(model, processor, prompt, max_tokens, view), flush=True)
