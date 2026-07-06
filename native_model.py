#!/usr/bin/env python3
"""Native GPU generation runner for iLLaDA / W1 experiments.

This replaces the OpenCompass inference path for mechanism-oriented tests.  It
reads the same `test_config.yaml`, reuses shared RULER prepared files and original benchmark data sources, and writes model outputs under:

    model_outputs/<model_alias>/<task>/<benchmark_alias>/<condition>/

Per-model generation manifests are written under:

    model_outputs/<model_alias>/native_model_manifest.{jsonl,csv}

Important efficiency property: the model is loaded once per selected model, then
all selected sweep conditions are generated in the same process.  This avoids the
old repeated load cost across gen_steps/threshold/config sweeps.

Supported backends:
  - illada: internal backend key for iLLaDA native masked-diffusion generation via generate.py
  - hf_causal / hf: HuggingFace AutoModelForCausalLM.generate
  - w1_4b / whale4b: local WhaleTech W1-4B dLLM release via SamplingRunner
  - openai_compatible / api: POST /chat/completions style local/API model

If the config has no top-level `models:` block, the legacy top-level `model:`
block is treated as an iLLaDA model named by `abbr` or `iLLaDA`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from run_test import ROOT, as_list, collect_experiments, deep_merge, expand_matrix, load_yaml, safe_name
except Exception:  # pragma: no cover
    ROOT = Path(__file__).resolve().parent
    from prepare_data import as_list, deep_merge, load_yaml, safe_name, collect_experiments, expand_matrix  # type: ignore

from prepare_data import bench_alias, condition_name, prepared_dir_for, prepared_file_for, read_jsonl, write_jsonl, json_default


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
        rows = [{"status": "empty"}]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def safe_model_name(value: str) -> str:
    """Filesystem-safe model name that preserves canonical casing.

    We keep iLLaDA as `iLLaDA` in model_outputs/native_outputs while still
    accepting lowercase CLI/config aliases such as `illada`.
    """
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_")
    text = re.sub(r"_+", "_", text)
    return text or "model"


def model_alias(model_cfg: Dict[str, Any]) -> str:
    return safe_model_name(str(model_cfg.get("name") or model_cfg.get("abbr") or Path(str(model_cfg.get("path", "model"))).name or "model"))


def model_keys(model_cfg: Dict[str, Any]) -> Set[str]:
    """All names that should select this model.

    This lets config use the canonical `iLLaDA` while old commands such as
    `--models illada` continue to work.
    """
    keys: Set[str] = set()
    for value in (model_cfg.get("name"), model_cfg.get("abbr"), model_cfg.get("backend"), model_alias(model_cfg)):
        if value is None:
            continue
        value = str(value)
        keys.add(value)
        keys.add(value.lower())
        keys.add(safe_name(value))
        keys.add(safe_model_name(value))
    return {k for k in keys if k}


def model_selected(model_cfg: Dict[str, Any], selected: Optional[Set[str]]) -> bool:
    if not selected:
        return True
    wanted: Set[str] = set()
    for item in selected:
        item = str(item)
        wanted.add(item)
        wanted.add(item.lower())
        wanted.add(safe_name(item))
        wanted.add(safe_model_name(item))
    return bool(model_keys(model_cfg) & wanted)


def normalize_models(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = config.get("models")
    models: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                models.append({"name": item, "backend": item})
            elif isinstance(item, dict):
                models.append(deepcopy(item))
    elif isinstance(raw, dict):
        for name, item in raw.items():
            cfg = deepcopy(item or {})
            cfg.setdefault("name", name)
            models.append(cfg)
    if not models:
        legacy = deepcopy(config.get("model", {}) or {})
        legacy.setdefault("name", legacy.get("abbr", "iLLaDA"))
        legacy.setdefault("backend", "illada")
        if legacy.get("type") in {"instruct", "base"}:
            legacy.setdefault("backend", "illada")
        models.append(legacy)
    for cfg in models:
        cfg.setdefault("name", model_alias(cfg))
        if "backend" not in cfg:
            cfg["backend"] = "illada" if cfg.get("type") in {"instruct", "base"} else "hf_causal"
    return models


def iter_conditions(config: Dict[str, Any], selected: Set[str]) -> Iterable[Dict[str, Any]]:
    defaults = config.get("defaults", {}) or {}
    for experiment in collect_experiments(config, selected):
        task = experiment.get("task") or "runs"
        exp_models = set(str(x) for x in as_list(experiment.get("models"))) if experiment.get("models") is not None else None
        for benchmark in as_list(experiment.get("benchmark")):
            for idx, params in enumerate(expand_matrix(experiment), start=1):
                merged = deep_merge(defaults, params)
                yield {
                    "task": task,
                    "experiment": experiment.get("name"),
                    "benchmark": str(benchmark),
                    "params": merged,
                    "condition": condition_name(merged),
                    "condition_index": idx,
                    "experiment_models": exp_models,
                }


def output_dir_for(root: Path, model_name: str, condition: Dict[str, Any]) -> Path:
    return root / safe_model_name(model_name) / safe_name(condition["task"] or "runs") / bench_alias(condition["benchmark"]) / condition["condition"]


def write_model_manifests(output_root: Path, stem: str, rows: List[Dict[str, Any]], write_global: bool = False) -> List[Path]:
    """Write manifests under each model directory.

    The run artifacts are model-first, so the manifest should be model-first as
    well.  By default we intentionally do not write output_root/<stem>.* because
    that puts a cross-model file next to model directories and makes long-running
    result roots hard to reason about.  Set write_global=True to also emit an
    aggregate copy under output_root/_manifests/.
    """
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        alias = safe_model_name(str(row.get("model") or "model"))
        by_model.setdefault(alias, []).append(row)

    written: List[Path] = []
    for alias, model_rows in sorted(by_model.items()):
        model_root = output_root / alias
        jsonl_path = model_root / f"{stem}.jsonl"
        csv_path = model_root / f"{stem}.csv"
        write_jsonl(jsonl_path, model_rows)
        write_csv(csv_path, model_rows)
        written.extend([jsonl_path, csv_path])

    if write_global:
        manifest_root = output_root / "_manifests"
        jsonl_path = manifest_root / f"{stem}.jsonl"
        csv_path = manifest_root / f"{stem}.csv"
        write_jsonl(jsonl_path, rows)
        write_csv(csv_path, rows)
        written.extend([jsonl_path, csv_path])

    return written


def prepared_input_path(config: Dict[str, Any], condition: Dict[str, Any]) -> Path:
    """Return shared prepared path.

    RULER synthetic data follows the original LLaDA/OpenCompass layout:
      data/prepared/<ruler_benchmark>/<condition>.jsonl
    Standard datasets (GSM8K/MBPP/custom_math) are not materialized here;
    the returned path is only a readable placeholder for dry-run messages.
    """
    data_cfg = config.get("data", {}) or {}
    base = Path(data_cfg.get("prepared_dir", "data/prepared"))
    if not base.is_absolute():
        base = ROOT / base
    benchmark = str(condition["benchmark"])
    if benchmark in {"ruler_niah_single_1", "ruler_niah_order_2", "ruler_niah_double_2"}:
        return prepared_file_for(base, benchmark, condition["params"])
    return base / benchmark / "original_source"


def load_existing_or_prepared_inputs(config: Dict[str, Any], condition: Dict[str, Any], input_path: Path) -> Tuple[List[Dict[str, Any]], Optional[Path], str]:
    """Load inputs for one condition.

    RULER variants are read from shared data/prepared/<benchmark>/<file>.jsonl.
    GSM8K / MBPP / custom_math reuse the original dataset sources rather than
    duplicating data into prepared directories.
    """
    benchmark = str(condition.get("benchmark"))

    if benchmark in {"ruler_niah_single_1", "ruler_niah_order_2", "ruler_niah_double_2"}:
        if input_path.exists():
            return read_jsonl(input_path), input_path, str(input_path)
        try:
            import prepare_data as prep
        except Exception as exc:
            raise SystemExit(f"Missing prepared RULER file {input_path} and failed to import prepare_data: {exc}") from exc
        data_cfg = config.get("data", {}) or {}
        haystack = Path(data_cfg.get("ruler_haystack_path", "data/ruler/paul_graham_essay.jsonl"))
        if not haystack.is_absolute():
            haystack = ROOT / haystack
        if benchmark == "ruler_niah_single_1":
            return prep.prepare_ruler_single(condition, haystack), None, "fallback:ruler_niah_single_1"
        return prep.prepare_ruler_order2(condition, haystack), None, f"fallback:{benchmark}"

    # Standard datasets: keep using the original source/caches.
    try:
        import prepare_data as prep
    except Exception as exc:
        raise SystemExit(f"Failed to import dataset loaders from prepare_data.py: {exc}") from exc

    if benchmark == "gsm8k":
        return prep.prepare_gsm8k(condition), None, "original:gsm8k/test"
    if benchmark == "mbpp":
        return prep.prepare_mbpp(condition), None, "original:mbpp/test"
    if benchmark == "custom_math":
        return prep.prepare_custom_math(condition, ROOT / "data"), None, "original:data/custom_math"

    raise SystemExit(f"Unsupported benchmark `{benchmark}` in native_model.py")


class GpuTelemetry:
    def __init__(self, path: Path, interval: float = 2.0):
        self.path = path
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "index", "name", "memory_used_mb", "memory_total_mb", "utilization_gpu", "power_draw_w"])
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval * 2, 1.0))

    def _loop(self) -> None:
        query = "index,name,memory.used,memory.total,utilization.gpu,power.draw"
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                now = utc_now()
                with self.path.open("a", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    for line in out.splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        writer.writerow([now] + parts)
            except Exception:
                pass
            self._stop.wait(self.interval)


class GenerationResult:
    def __init__(self, text: str, elapsed: float, tokens_generated: Optional[int] = None, trace: Optional[Dict[str, Any]] = None):
        self.text = text
        self.elapsed = elapsed
        self.tokens_generated = tokens_generated
        self.trace = trace


class BaseAdapter:
    supports_trace = False

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.alias = model_alias(cfg)

    def generate_one(self, prompt: str, params: Dict[str, Any]) -> GenerationResult:
        raise NotImplementedError

    def close(self) -> None:
        pass


class ILLADAAdapter(BaseAdapter):
    supports_trace = True

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        import torch
        from transformers import AutoModel, AutoTokenizer
        from generate import generate

        self.torch = torch
        self.generate_fn = generate
        self.device = str(cfg.get("device", "cuda"))
        path = str(cfg.get("path", "GSAI-ML/iLLaDA-8B-Instruct"))
        model_kwargs = deepcopy(cfg.get("model_kwargs", {}) or {})
        dtype_value = cfg.get("torch_dtype", cfg.get("dtype"))
        if dtype_value is not None and "torch_dtype" not in model_kwargs:
            model_kwargs["torch_dtype"] = self._parse_torch_dtype(dtype_value)
        elif isinstance(model_kwargs.get("torch_dtype"), str):
            model_kwargs["torch_dtype"] = self._parse_torch_dtype(model_kwargs["torch_dtype"])
        if "torch_dtype" not in model_kwargs:
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True, **model_kwargs)
        if not model_kwargs.get("device_map"):
            self.model = self.model.to(self.device)
        self.model.eval()
        self.mask_id = int(cfg.get("mask_id", 5))
        self.apply_chat_template = bool(cfg.get("apply_chat_template", True))
        self.path = path

    def _parse_torch_dtype(self, value: Any):
        if not isinstance(value, str):
            return value
        value = value.replace("torch.", "")
        return getattr(self.torch, value)

    @property
    def model_device(self):
        try:
            return self.model.device
        except Exception:
            return next(self.model.parameters()).device

    def _render_prompt(self, prompt: str) -> str:
        if self.apply_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        return prompt

    def _trim_output(self, text: str, params: Dict[str, Any]) -> str:
        stops = []
        stops.extend(as_list(params.get("until")))
        stops.extend(as_list(self.cfg.get("until")))
        for tok_attr in ("eos_token", "pad_token"):
            tok = getattr(self.tokenizer, tok_attr, None)
            if tok:
                stops.append(tok)
        stops.extend(["<|eot_id|>", "<|endoftext|>"])
        for stop in stops:
            if stop and stop in text:
                text = text.split(str(stop))[0]
        ids = self.tokenizer(text, add_special_tokens=False).get("input_ids", [])
        try:
            return self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        except Exception:
            return text.strip()

    def generate_one(self, prompt: str, params: Dict[str, Any]) -> GenerationResult:
        torch = self.torch
        rendered = self._render_prompt(prompt)
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.model_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model_device)

        if self.model_device.type == "cuda":
            torch.cuda.synchronize(self.model_device)
            torch.cuda.reset_peak_memory_stats(self.model_device)
        started = time.perf_counter()
        with torch.no_grad():
            generated = self.generate_fn(
                self.model,
                input_ids,
                attention_mask=attention_mask,
                steps=int(params.get("gen_steps", params.get("steps", params.get("gen_length", 128)))),
                gen_length=int(params.get("gen_length", 128)),
                block_length=int(params.get("gen_blocksize", params.get("block_length", params.get("gen_length", 128)))),
                temperature=float(params.get("temperature", self.cfg.get("temperature", 0.0)) or 0.0),
                cfg_scale=float(params.get("cfg", self.cfg.get("cfg", 0.0)) or 0.0),
                remasking=str(params.get("remasking", self.cfg.get("remasking", "low_confidence"))),
                mask_id=int(params.get("mask_id", self.mask_id)),
                logits_eos_inf=bool(params.get("diff_logits_eos_inf", params.get("logits_eos_inf", False))),
                confidence_eos_eot_inf=bool(params.get("diff_confidence_eos_eot_inf", params.get("confidence_eos_eot_inf", False))),
                token_selection_confidence_threshold=params.get("token_selection_confidence_threshold"),
                min_transfer_tokens=int(params.get("min_transfer_tokens", 1)),
                return_trace=bool(params.get("return_trace", False)),
                trace_token_snapshots=bool(params.get("trace_token_snapshots", False) or params.get("trace_decode_snapshots", False)),
                tokenizer=self.tokenizer,
                speed_schedule_name=params.get("speed_schedule_name"),
                steps_per_block_schedule=params.get("steps_per_block_schedule"),
                token_selection_confidence_threshold_schedule=params.get("token_selection_confidence_threshold_schedule"),
                trace_step0_full_confidence=bool(params.get("trace_step0_full_confidence", False)),
                decode_order=str(params.get("decode_order", "confidence")),
            )
        if self.model_device.type == "cuda":
            torch.cuda.synchronize(self.model_device)
        elapsed = time.perf_counter() - started
        if bool(params.get("return_trace", False)):
            token_tensor, trace = generated
        else:
            token_tensor, trace = generated, None
        answer_ids = token_tensor[0][input_ids.shape[1]:]
        raw_text = self.tokenizer.decode(answer_ids, skip_special_tokens=False)
        text = self._trim_output(raw_text, params)
        return GenerationResult(text=text, elapsed=elapsed, tokens_generated=int(params.get("gen_length", 128)), trace=trace)


class HFCausalAdapter(BaseAdapter):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = str(cfg.get("device", "cuda"))
        path = str(cfg.get("path") or cfg.get("model") or cfg.get("name"))
        model_kwargs = deepcopy(cfg.get("model_kwargs", {}) or {})
        dtype_value = cfg.get("torch_dtype", cfg.get("dtype"))
        if dtype_value and "torch_dtype" not in model_kwargs:
            model_kwargs["torch_dtype"] = getattr(torch, str(dtype_value).replace("torch.", "")) if isinstance(dtype_value, str) else dtype_value
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, **model_kwargs)
        if not model_kwargs.get("device_map"):
            self.model = self.model.to(self.device)
        self.model.eval()
        self.apply_chat_template = bool(cfg.get("apply_chat_template", True))
        self.path = path

    @property
    def model_device(self):
        try:
            return self.model.device
        except Exception:
            return next(self.model.parameters()).device

    def render_prompt(self, prompt: str) -> str:
        if self.apply_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        return prompt

    def generate_one(self, prompt: str, params: Dict[str, Any]) -> GenerationResult:
        torch = self.torch
        rendered = self.render_prompt(prompt)
        encoded = self.tokenizer(rendered, return_tensors="pt").to(self.model_device)
        started = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **encoded,
                max_new_tokens=int(params.get("max_new_tokens", params.get("gen_length", 128))),
                do_sample=bool(float(params.get("temperature", 0.0) or 0.0) > 0),
                temperature=max(float(params.get("temperature", 0.0) or 0.0), 1e-6),
                pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )
        elapsed = time.perf_counter() - started
        answer = self.tokenizer.decode(out[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return GenerationResult(text=answer, elapsed=elapsed, tokens_generated=int(out.shape[1] - encoded["input_ids"].shape[1]))



class W14BLocalAdapter(BaseAdapter):
    """Local WhaleTech W1-4B dLLM backend.

    This adapter uses the public W1-4B release package directly, not an
    OpenAI-compatible server.  It instantiates ``SamplingRunner`` once and then
    mutates ``runner.cfg`` per condition so the 7.48GB checkpoint is not
    reloaded for every steps/sampler/context sweep.

    Expected config keys:
      repo_path: local clone path of WhaletechAI/W1-4B-dLLM-Base
      checkpoint: optional; defaults to repo_path/whale3.7Bdiffusion.safetensors
      config: optional; defaults to repo_path/configs/whale3b.yaml
      tokenizer_path: optional; defaults to repo_path/whale-tokenizer
      sampler: standard | gidd | jump
      dtype: bf16 | fp16 | fp32
      device: cuda | cpu
    """
    supports_trace = True

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        import importlib
        import importlib.util
        import torch

        self.torch = torch
        repo_value = str(cfg.get("repo_path") or cfg.get("path") or os.getenv("W1_4B_REPO", "W1-4B-dLLM-Base"))
        repo_value = self._expand_env_default(repo_value)
        self.repo_path = Path(repo_value).expanduser().resolve()
        if not self.repo_path.exists():
            raise SystemExit(
                f"W1 repo_path not found: {self.repo_path}\n"
                "Clone it first, for example:\n"
                "  git clone https://huggingface.co/WhaletechAI/W1-4B-dLLM-Base\n"
                "Then set models.w1_4b.repo_path in test_config_native.yaml."
            )
        self._ensure_whale4b_import(importlib, importlib.util)
        from whale4b.core.runner import RunConfig, SamplingRunner

        ckpt = Path(str(cfg.get("checkpoint") or self.repo_path / "whale3.7Bdiffusion.safetensors")).expanduser()
        config_path = Path(str(cfg.get("config") or self.repo_path / "configs" / "whale3b.yaml")).expanduser()
        tokenizer_path = Path(str(cfg.get("tokenizer_path") or self.repo_path / "whale-tokenizer")).expanduser()
        if not ckpt.is_absolute():
            ckpt = (self.repo_path / ckpt).resolve()
        if not config_path.is_absolute():
            config_path = (self.repo_path / config_path).resolve()
        if not tokenizer_path.is_absolute():
            tokenizer_path = (self.repo_path / tokenizer_path).resolve()
        missing = [str(x) for x in [ckpt, config_path, tokenizer_path] if not x.exists()]
        if missing:
            raise SystemExit("W1 local backend is missing required file(s):\n" + "\n".join(missing))

        self.device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = str(cfg.get("dtype", "bf16"))
        self.max_seq_len = int(cfg.get("max_seq_len", 4096))
        run_cfg = RunConfig(
            ckpt_path=str(ckpt),
            config_path=str(config_path),
            tokenizer_path=str(tokenizer_path),
            sampler=str(cfg.get("sampler", "gidd")),
            steps=int(cfg.get("steps", 64)),
            max_new_tokens=int(cfg.get("max_new_tokens", 256)),
            temperature=float(cfg.get("temperature", 0.0) or 0.0),
            top_k=int(cfg.get("top_k", 0) or 0),
            device=self.device,
            dtype=self.dtype,
            seed=cfg.get("seed", 1234),
            use_ema=bool(cfg.get("use_ema", True)),
            strict=bool(cfg.get("strict", False)),
        )
        # Optional W1 sampler-specific defaults.
        for key in [
            "p", "jump_last_steps", "jump_frac", "jump_min_tokens", "no_mask_jump",
            "gidd_eps", "gidd_min_p", "posterior_temperature", "suppress_mask_clean",
            "rho_mode", "gidd_exact_mode", "fail_on_negative_mass",
        ]:
            if key in cfg:
                setattr(run_cfg, key, cfg[key])
        self.runner = SamplingRunner(run_cfg)
        # Expose W1 tokenizer so trace artifacts can decode token ids instead of
        # writing raw ids or mask tokens into token timelines.
        self.tokenizer = getattr(self.runner, "tokenizer", None)
        self.path = str(self.repo_path)

    def _expand_env_default(self, value: str) -> str:
        # Supports simple shell-like forms used in YAML, e.g.
        # ${W1_4B_REPO:-W1-4B-dLLM-Base} and ${W1_4B_REPO}.
        m = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", value.strip())
        if m:
            env_value = os.getenv(m.group(1))
            if env_value:
                return env_value
            return m.group(2) or ""
        return os.path.expandvars(value)

    def _ensure_whale4b_import(self, importlib, importlib_util) -> None:
        """Import the HF repo even when the local folder is not named whale4b."""
        try:
            importlib.import_module("whale4b")
            return
        except Exception:
            pass
        # Common case after `git clone ... W1-4B-dLLM-Base`: the repo root is the
        # package content but not named whale4b.  Load it under the whale4b name.
        init_py = self.repo_path / "__init__.py"
        if not init_py.exists():
            raise SystemExit(f"Cannot import whale4b and {init_py} does not exist.")
        parent = str(self.repo_path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        spec = importlib_util.spec_from_file_location(
            "whale4b", str(init_py), submodule_search_locations=[str(self.repo_path)]
        )
        if spec is None or spec.loader is None:
            raise SystemExit(f"Could not create import spec for W1 repo: {self.repo_path}")
        module = importlib_util.module_from_spec(spec)
        sys.modules["whale4b"] = module
        spec.loader.exec_module(module)

    def _update_runner_cfg(self, params: Dict[str, Any]) -> None:
        cfg = self.runner.cfg
        cfg.sampler = str(params.get("w1_sampler", params.get("sampler", self.cfg.get("sampler", cfg.sampler))))
        cfg.steps = int(params.get("w1_steps", params.get("gen_steps", params.get("steps", cfg.steps))))
        cfg.max_new_tokens = int(params.get("max_new_tokens", params.get("gen_length", cfg.max_new_tokens)))
        cfg.temperature = float(params.get("temperature", self.cfg.get("temperature", cfg.temperature)) or 0.0)
        cfg.top_k = int(params.get("top_k", self.cfg.get("top_k", cfg.top_k)) or 0)
        for key in [
            "p", "jump_last_steps", "jump_frac", "jump_min_tokens", "no_mask_jump",
            "gidd_eps", "gidd_min_p", "posterior_temperature", "suppress_mask_clean",
            "rho_mode", "gidd_exact_mode", "fail_on_negative_mass",
        ]:
            if key in params and params[key] is not None:
                setattr(cfg, key, params[key])

    def _decode_token_id(self, token_id: int) -> str:
        tok = getattr(self, "tokenizer", None)
        if tok is not None:
            try:
                return tok.decode([int(token_id)], skip_special_tokens=False)
            except TypeError:
                try:
                    return tok.decode([int(token_id)])
                except Exception:
                    pass
            except Exception:
                pass
        return str(token_id)

    def _make_trace_callback(self):
        torch = self.torch
        rows: List[Dict[str, Any]] = []
        prev_x = {"value": None}

        def callback(state):
            x = state.x_t
            prefix_len = int(state.prefix_len)
            mask_id = int(state.mask_token_id)
            editable = torch.ones_like(x, dtype=torch.bool)
            editable[:, :prefix_len] = False

            if prev_x["value"] is None:
                # First callback: count positions that are already visible in the
                # generated span, but do not treat still-masked positions as
                # committed tokens.
                changed = x.ne(mask_id) & editable
                changed_any = x.ne(mask_id) & editable
            else:
                changed_any = x.ne(prev_x["value"]) & editable
                # For the token timeline, keep only non-mask updates.  W1 can
                # revise/remask positions; if we record mask updates as selected
                # tokens, the visualizer shows literal [MASK] instead of the old
                # box/commit-style view.
                changed = changed_any & x.ne(mask_id)

            abs_nonmask = torch.where(changed[0])[0].tolist() if changed.numel() else []
            abs_any = torch.where(changed_any[0])[0].tolist() if changed_any.numel() else []
            abs_mask = [int(p) for p in abs_any if int(p) >= prefix_len and int(x[0, int(p)].item()) == mask_id]

            selected_positions = [int(p - prefix_len) for p in abs_nonmask if int(p) >= prefix_len]
            selected_token_ids = [int(x[0, p].item()) for p in abs_nonmask if int(p) >= prefix_len]
            changed_positions = [int(p - prefix_len) for p in abs_any if int(p) >= prefix_len]
            mask_positions = [int(p - prefix_len) for p in abs_mask]

            # Full generated-span state for compact visual inspection.  Masks are
            # represented as boxes rather than literal [MASK] strings:
            #   ■ = currently visible token, □ = still masked position.
            try:
                generated_span_ids = [int(t) for t in x[0, prefix_len:].detach().cpu().tolist()]
            except Exception:
                generated_span_ids = []
            state_boxes = _boxes_from_token_ids(generated_span_ids, mask_id)
            visible_positions = [i for i, tid in enumerate(generated_span_ids) if int(tid) != mask_id]

            rows.append({
                "step_idx": int(state.step),
                "block_idx": 0,
                "t": float(state.t),
                "mask_count_before": int(state.remain_before),
                "mask_count_after": int(state.remain_after),
                "selected_positions": selected_positions,
                "selected_token_ids": selected_token_ids,
                "selected_decoded_tokens": [self._decode_token_id(t) for t in selected_token_ids],
                "selected_confidences": [],
                "changed_positions": changed_positions,
                "changed_count": len(changed_positions),
                "mask_positions": mask_positions,
                "mask_update_count": len(mask_positions),
                "visible_positions": visible_positions,
                "visible_count": len(visible_positions),
                "state_boxes": state_boxes,
                "state_boxes_compact": _compact_boxes(state_boxes),
                "mask_boxes": _compact_boxes(_mask_boxes(state.remain_after)),
                "transfer_reason": "w1_sampler",
                "cumulative_transferred_tokens": None,
                "current_completion_rate": None,
                "w1_selected_count": int(state.selected),
                "w1_metadata": dict(state.metadata or {}),
            })
            prev_x["value"] = x.clone()

        return rows, callback

    def generate_one(self, prompt: str, params: Dict[str, Any]) -> GenerationResult:
        self._update_runner_cfg(params)
        # W1 config has max_seq_len=4096.  We do not truncate here; failing loudly
        # is preferable to silently changing context-test difficulty.
        want_trace = bool(params.get("return_trace", False))
        trace_rows, callback = self._make_trace_callback() if want_trace else ([], None)
        result = self.runner.run(prompt=prompt, callback=callback)
        trace = None
        if want_trace:
            total_selected = sum(len(r.get("selected_positions") or []) for r in trace_rows)
            steps = max(1, len(trace_rows))
            gen_tokens = int(getattr(result, "generated_tokens", params.get("max_new_tokens", params.get("gen_length", 0))) or 0)
            trace = {
                "backend": "w1_4b",
                "sampler": getattr(result, "sampler", self.runner.cfg.sampler),
                "steps_run": int(getattr(result, "steps_run", self.runner.cfg.steps)),
                "generated_tokens": gen_tokens,
                "step_stats": trace_rows,
                "completion_rate": 1.0,
                "actual_parallelism": float(total_selected / steps) if steps else None,
                "actual_arness": None,
                "threshold_pass_rate": None,
                "fallback_rate": None,
            }
        return GenerationResult(
            text=str(getattr(result, "new_text", "")).strip(),
            elapsed=float(getattr(result, "elapsed_s", 0.0)),
            tokens_generated=int(getattr(result, "generated_tokens", params.get("max_new_tokens", params.get("gen_length", 0))) or 0),
            trace=trace,
        )


class OpenAICompatibleAdapter(BaseAdapter):
    """OpenAI-compatible chat backend, including WhaleTech W1-style local APIs.

    The base OpenAI schema is deliberately kept simple, while model-specific
    decoding knobs can be forwarded through config-driven mappings.  This lets
    the same native pipeline test W1-style parallel decoding without changing
    this file again.

    Config keys supported:
      generation_params: static fields merged into every request payload.
      extra_body: static vendor-specific object merged into payload["extra_body"].
      api_extra_params: list of experiment params copied into payload["extra_body"].
      api_param_map: mapping from experiment param -> extra_body key.
      pass_params_to_extra_body: if true, copy all safe experiment params.

    Example:
      api_param_map:
        w1_parallel_tokens: parallel_tokens
        w1_decode_order: decode_order
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.api_base = str(cfg.get("api_base") or os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = str(cfg.get("api_key") or os.getenv("OPENAI_API_KEY", "EMPTY"))
        self.model_name = str(cfg.get("model") or cfg.get("path") or cfg.get("name"))

    def _extra_body_from_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        extra: Dict[str, Any] = deepcopy(self.cfg.get("extra_body", {}) or {})
        for key in self.cfg.get("api_extra_params", []) or []:
            if key in params and params[key] is not None:
                extra[str(key)] = params[key]
        for src, dst in (self.cfg.get("api_param_map", {}) or {}).items():
            if src in params and params[src] is not None:
                extra[str(dst)] = params[src]
        if self.cfg.get("pass_params_to_extra_body", False):
            skip = {"gen_length", "max_new_tokens", "temperature", "sample_limit", "sample_indices", "num_samples",
                    "return_trace", "trace_token_snapshots", "trace_decode_snapshots", "context_length",
                    "needle_position", "needle_pair"}
            for key, value in params.items():
                if key not in skip and value is not None and isinstance(value, (str, int, float, bool, list, dict)):
                    extra.setdefault(str(key), value)
        return extra

    def generate_one(self, prompt: str, params: Dict[str, Any]) -> GenerationResult:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(params.get("temperature", 0.0) or 0.0),
            "max_tokens": int(params.get("max_new_tokens", params.get("gen_length", 128))),
        }
        payload.update(deepcopy(self.cfg.get("generation_params", {}) or {}))
        extra_body = self._extra_body_from_params(params)
        if extra_body:
            # Many OpenAI-compatible servers accept this field (vLLM style).
            # If your W1 server expects top-level vendor fields instead, set
            # config: extra_body_as_top_level: true.
            if self.cfg.get("extra_body_as_top_level", False):
                payload.update(extra_body)
            else:
                payload["extra_body"] = extra_body

        data = json.dumps(payload, default=json_default).encode("utf-8")
        req = urllib.request.Request(
            self.api_base + "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=float(self.cfg.get("timeout", 600))) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"API request failed: {exc.code} {body[:500]}") from exc
        elapsed = time.perf_counter() - started
        choice = obj["choices"][0]
        text = (choice.get("message", {}) or {}).get("content")
        if text is None:
            text = choice.get("text", "")
        text = str(text).strip()
        usage = obj.get("usage", {}) or {}
        return GenerationResult(text=text, elapsed=elapsed, tokens_generated=usage.get("completion_tokens"))


def build_adapter(cfg: Dict[str, Any]) -> BaseAdapter:
    backend = str(cfg.get("backend", "illada")).lower()
    if backend in {"illada", "llada", "diffusion"}:
        return ILLADAAdapter(cfg)
    if backend in {"hf", "hf_causal", "transformers", "causal"}:
        return HFCausalAdapter(cfg)
    if backend in {"w1_4b", "whale4b", "w1_local", "w1hf"}:
        return W14BLocalAdapter(cfg)
    if backend in {"openai_compatible", "api", "w1", "vllm"}:
        return OpenAICompatibleAdapter(cfg)
    raise SystemExit(f"Unsupported model backend `{backend}` for model {cfg.get('name')}")


TRACE_MASK_RENDERING = "square_uncompressed"


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        s = str(value).strip()
        if s == "" or s.lower() in {"none", "null", "nan"}:
            return default
        return int(float(s))
    except Exception:
        return default


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        s = str(value).strip()
        if s == "" or s.lower() in {"none", "null", "nan"}:
            return default
        return float(s)
    except Exception:
        return default


def _mean(values: Sequence[Any]) -> Optional[float]:
    vals = []
    for value in values:
        v = _to_float(value, None)
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def _expand_mask_runs(text: Any) -> str:
    """Render all literal/compressed mask placeholders as one □ per mask."""
    if text is None:
        return ""
    s = str(text)

    def repl(m: re.Match[str]) -> str:
        n = _to_int(m.group(1), 1)
        return "□" * max(1, n)

    # Normalize both old literal trace forms and already-boxed compressed forms.
    s = re.sub(r"\[MASK\]\s*[x×]\s*(\d+)", repl, s, flags=re.IGNORECASE)
    s = re.sub(r"□\s*[x×]\s*(\d+)", repl, s)
    s = re.sub(r"\[MASK\]", "□", s, flags=re.IGNORECASE)
    return s


def _sanitize_mask_obj(obj: Any) -> Any:
    """Recursively normalize mask placeholders in trace artifacts only."""
    if isinstance(obj, str):
        return _expand_mask_runs(obj)
    if isinstance(obj, list):
        return [_sanitize_mask_obj(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_mask_obj(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _sanitize_mask_obj(v) for k, v in obj.items()}
    return obj


def _json_cell(value: Any) -> str:
    """CSV cell helper: keep list/dict columns parseable by visualizers."""
    return json.dumps(_sanitize_mask_obj(value), ensure_ascii=False, default=json_default)


VISIBLE_BOX = "■"
MASK_BOX = "□"


def _compact_boxes(boxes: str, group: int = 16) -> str:
    """Group a long mask/visible box line so CSV/HTML is compact and readable."""
    raw = "".join(ch for ch in str(boxes) if ch in {VISIBLE_BOX, MASK_BOX})
    if not raw:
        return ""
    group = max(1, int(group or 16))
    return " ".join(raw[i:i + group] for i in range(0, len(raw), group))


def _ungroup_boxes(boxes: Any) -> str:
    return "".join(ch for ch in str(boxes or "") if ch in {VISIBLE_BOX, MASK_BOX})


def _boxes_from_token_ids(token_ids: Sequence[Any], mask_id: int) -> str:
    chars: List[str] = []
    for token_id in token_ids:
        try:
            tid = int(token_id)
        except Exception:
            chars.append(VISIBLE_BOX)
            continue
        chars.append(MASK_BOX if tid == int(mask_id) else VISIBLE_BOX)
    return "".join(chars)


def _mask_boxes(count: Any) -> str:
    try:
        n = max(0, int(count))
    except Exception:
        n = 0
    return MASK_BOX * n


def _decode_ids(tokenizer: Any, token_ids: Sequence[Any], mask_id: Optional[int] = None) -> List[str]:
    decoded: List[str] = []
    for token_id in token_ids:
        try:
            tid = int(token_id)
        except Exception:
            decoded.append(str(token_id))
            continue
        if mask_id is not None and tid == int(mask_id):
            # Do not put literal [MASK] tokens into the commit timeline.  The
            # timeline should display committed/visible tokens; masks are tracked
            # separately through mask_count_* and mask_positions.
            continue
        if tokenizer is not None:
            try:
                decoded.append(tokenizer.decode([tid], skip_special_tokens=False))
                continue
            except TypeError:
                try:
                    decoded.append(tokenizer.decode([tid]))
                    continue
                except Exception:
                    pass
            except Exception:
                pass
        decoded.append(str(tid))
    return decoded


def _as_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for x in value:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out
    return []


def _threshold_label(value: Any) -> str:
    if value is None:
        return "none"
    s = str(value).strip().lower()
    if s in {"", "none", "null", "nan"}:
        return "none"
    return s.replace(".", "p")


def _render_trace_slots(slots: Sequence[Optional[str]], max_chars: int = 0) -> str:
    """Render a generation-chain block with visible tokens and □ for masks."""
    s = "".join(MASK_BOX if tok is None else _expand_mask_runs(tok) for tok in slots)
    s = s.replace("\r", "\\r").replace("\n", "\\n")
    if max_chars > 0 and len(s) > max_chars:
        return s[:max_chars] + "...<truncated>"
    return s


def _trace_local_pos(pos: int, block_idx: int, block_len: int) -> Optional[int]:
    start = block_idx * block_len
    if start <= pos < start + block_len:
        return pos - start
    if 0 <= pos < block_len:
        return pos
    return None


def _add_generation_chain_block_columns(
    out: Dict[str, Any],
    block_slots: Dict[int, List[Optional[str]]],
    num_blocks: int,
) -> None:
    for b in range(num_blocks):
        visible = sum(x is not None for x in block_slots[b])
        total = len(block_slots[b])
        out[f"block_{b:02d}"] = _render_trace_slots(block_slots[b])
        out[f"block_{b:02d}_complete"] = f"{visible}/{total}"


def _build_generation_chain_rows(
    rows: List[Dict[str, Any]],
    gen_length: int,
    block_len: int,
    num_blocks: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Build export_trace-style generation_chain.csv rows directly in native_model."""
    block_slots: Dict[int, List[Optional[str]]] = {}
    for b in range(num_blocks):
        start = b * block_len
        if gen_length:
            this_len = max(1, min(block_len, max(gen_length, block_len) - start))
        else:
            this_len = block_len
        block_slots[b] = [None] * this_len

    local_round_counter: Dict[int, int] = {b: 0 for b in range(num_blocks)}
    block_selected_counts: Dict[int, List[int]] = {b: [] for b in range(num_blocks)}
    block_confidences: Dict[int, List[float]] = {b: [] for b in range(num_blocks)}
    block_reasons: Dict[int, List[str]] = {b: [] for b in range(num_blocks)}

    chain_rows: List[Dict[str, Any]] = []
    initial: Dict[str, Any] = {
        "generation_step": -1,
        "active_block": "",
        "block_local_round": -1,
        "selected_count": 0,
        "transfer_reason": "initial_all_mask",
    }
    _add_generation_chain_block_columns(initial, block_slots, num_blocks)
    chain_rows.append(initial)

    for order, row in enumerate(rows):
        block_idx = _to_int(row.get("block_idx"), 0)
        block_idx = max(0, min(block_idx, num_blocks - 1))
        local_round = local_round_counter.get(block_idx, 0)
        local_round_counter[block_idx] = local_round + 1

        positions = _as_int_list(row.get("selected_positions"))
        tokens = row.get("selected_decoded_tokens") or []
        if not isinstance(tokens, list):
            tokens = [tokens]
        confidences = row.get("selected_confidences") or []
        reason = str(row.get("transfer_reason", ""))

        for pos, tok in zip(positions, tokens):
            lp = _trace_local_pos(_to_int(pos, -1), block_idx, block_len)
            if lp is not None and 0 <= lp < len(block_slots[block_idx]):
                block_slots[block_idx][lp] = _expand_mask_runs(tok)

        selected_count = len(positions)
        block_selected_counts.setdefault(block_idx, []).append(selected_count)
        for c in confidences if isinstance(confidences, list) else []:
            cf = _to_float(c, None)
            if cf is not None:
                block_confidences.setdefault(block_idx, []).append(cf)
        if reason:
            block_reasons.setdefault(block_idx, []).append(reason)

        out: Dict[str, Any] = {
            "generation_step": _to_int(row.get("step_idx"), order),
            "active_block": block_idx,
            "block_local_round": local_round,
            "selected_count": selected_count,
            "transfer_reason": reason,
        }
        _add_generation_chain_block_columns(out, block_slots, num_blocks)
        chain_rows.append(out)

    final_visible_by_block = {
        f"block_{b:02d}_final_complete": f"{sum(x is not None for x in block_slots[b])}/{len(block_slots[b])}"
        for b in range(num_blocks)
    }

    block_stats: List[Dict[str, Any]] = []
    for b in range(num_blocks):
        visible = sum(x is not None for x in block_slots[b])
        total = len(block_slots[b])
        reasons = block_reasons.get(b, [])
        block_stats.append({
            "block_idx": b,
            "complete": f"{visible}/{total}",
            "is_complete": visible == total,
            "local_rounds": local_round_counter.get(b, 0),
            "selected_tokens_total": sum(block_selected_counts.get(b, [])),
            "mean_selected_count": _mean(block_selected_counts.get(b, [])),
            "mean_confidence": _mean(block_confidences.get(b, [])),
            "fallback_steps": sum(1 for x in reasons if "fallback" in str(x)),
            "threshold_pass_steps": sum(1 for x in reasons if str(x) == "threshold_pass"),
        })

    return chain_rows, block_stats, final_visible_by_block


def _write_trace_metric_indexes(trace_root: Path, metrics: Dict[str, Any]) -> None:
    """Maintain sample_traces/trace_metrics.jsonl and .csv during native runs."""
    trace_root.mkdir(parents=True, exist_ok=True)
    metric_path = trace_root / "trace_metrics.jsonl"
    append_jsonl(metric_path, _sanitize_mask_obj(metrics))
    try:
        all_rows = read_jsonl(metric_path)
    except Exception:
        all_rows = [_sanitize_mask_obj(metrics)]
    write_csv(trace_root / "trace_metrics.csv", _sanitize_mask_obj(all_rows))


def _export_canonical_sample_trace(
    out_dir: Path,
    sample_dir: Path,
    condition: Dict[str, Any],
    sample: Dict[str, Any],
    result: GenerationResult,
    rows: List[Dict[str, Any]],
) -> None:
    """Write the canonical files expected by visual_arness_trace.py.

    The legacy visualizer expects step_events.csv, block_timeline.csv,
    block_metrics.csv and sample_metrics.json under sample_traces/sample_XXXX.
    The first native W1 patch only wrote trace.json; as a result, the visualizer
    fell back to raw trace rows and displayed literal mask tokens.  This export
    reconstructs a compact mask/visible view from selected_positions or native
    state_boxes so the HTML/plots show square boxes instead of literal [MASK].
    """
    params = condition.get("params", {}) or {}
    gen_length = int(params.get("gen_length", params.get("max_new_tokens", result.tokens_generated or 0)) or 0)
    block_len = int(params.get("gen_blocksize", gen_length or 1) or (gen_length or 1))
    block_len = max(1, block_len)
    num_blocks = max(1, (max(gen_length, 1) + block_len - 1) // block_len)

    # Cumulative commit state per block.  For W1 there is one logical generated
    # span/block.  For iLLaDA, selected_positions are usually block-local when
    # block_idx is present; the visualizer later converts to global positions.
    committed: Dict[int, set[int]] = {b: set() for b in range(num_blocks)}
    step_rows: List[Dict[str, Any]] = []
    timeline_rows: List[Dict[str, Any]] = []
    block_stats: Dict[int, Dict[str, Any]] = {b: {"selected_tokens_total": 0, "local_rounds": 0, "fallback_steps": 0, "conf": []} for b in range(num_blocks)}

    for order, row in enumerate(rows):
        step_idx = int(row.get("step_idx") if row.get("step_idx") is not None else order)
        block_idx = int(row.get("block_idx") if row.get("block_idx") is not None else 0)
        block_idx = max(0, min(block_idx, num_blocks - 1))
        positions = _as_int_list(row.get("selected_positions"))
        token_ids = _as_int_list(row.get("selected_token_ids"))
        decoded = row.get("selected_decoded_tokens") or []
        confs = row.get("selected_confidences") or []
        selected_count = len(positions)
        raw_state_boxes = _ungroup_boxes(row.get("state_boxes") or row.get("state_boxes_compact"))

        for pos in positions:
            if 0 <= int(pos) < block_len:
                committed.setdefault(block_idx, set()).add(int(pos))

        block_stats.setdefault(block_idx, {"selected_tokens_total": 0, "local_rounds": 0, "fallback_steps": 0, "conf": []})
        block_stats[block_idx]["selected_tokens_total"] += selected_count
        block_stats[block_idx]["local_rounds"] += 1
        if str(row.get("transfer_reason", "")).lower().startswith("fallback"):
            block_stats[block_idx]["fallback_steps"] += 1
        for c in confs if isinstance(confs, list) else []:
            try:
                block_stats[block_idx]["conf"].append(float(c))
            except Exception:
                pass

        step_rows.append({
            "generation_step": step_idx,
            "global_step_idx": step_idx,
            "active_block_idx": block_idx,
            "block_idx": block_idx,
            "block_local_round": block_stats[block_idx]["local_rounds"],
            "actual_transfer_count": selected_count,
            "selected_count": selected_count,
            "selected_positions": _json_cell(positions),
            "selected_token_ids": _json_cell(token_ids),
            "selected_decoded_tokens": _json_cell(decoded),
            "selected_confidences": _json_cell(confs),
            "mean_selected_confidence": (sum(block_stats[block_idx]["conf"]) / len(block_stats[block_idx]["conf"])) if block_stats[block_idx]["conf"] else "",
            "mask_count_before": row.get("mask_count_before"),
            "mask_count_after": row.get("mask_count_after"),
            "changed_positions": _json_cell(row.get("changed_positions") or []),
            "changed_count": row.get("changed_count", row.get("w1_selected_count", selected_count)),
            "mask_positions": _json_cell(row.get("mask_positions") or []),
            "mask_update_count": row.get("mask_update_count", 0),
            "mask_boxes": row.get("mask_boxes") or _compact_boxes(_mask_boxes(row.get("mask_count_after", 0))),
            "state_boxes": _compact_boxes(raw_state_boxes) if raw_state_boxes else "",
            "visible_count": row.get("visible_count", raw_state_boxes.count(VISIBLE_BOX) if raw_state_boxes else ""),
            "transfer_reason": row.get("transfer_reason", ""),
            "current_completion_rate": row.get("current_completion_rate", ""),
        })

        trow: Dict[str, Any] = {
            "generation_step": step_idx,
            "global_step_idx": step_idx,
            "active_block_idx": block_idx,
        }
        for b in range(num_blocks):
            # Last block can be shorter.
            start = b * block_len
            this_len = max(0, min(block_len, max(gen_length, block_len) - start)) if gen_length else block_len
            this_len = max(1, this_len)
            if raw_state_boxes:
                block_boxes = raw_state_boxes[start:start + this_len]
                if len(block_boxes) < this_len:
                    block_boxes += MASK_BOX * (this_len - len(block_boxes))
                n = block_boxes.count(VISIBLE_BOX)
            else:
                n = len(committed.get(b, set()))
                block_boxes = VISIBLE_BOX * min(n, this_len) + MASK_BOX * max(0, this_len - min(n, this_len))
            trow[f"block_{b}_complete"] = f"{min(n, this_len)}/{this_len}"
            # Human-readable square view for quick inspection in CSV/HTML.
            # It is intentionally box-only: no literal [MASK] token appears here.
            trow[f"block_{b}_boxes"] = _compact_boxes(block_boxes)
        timeline_rows.append(trow)

    block_rows: List[Dict[str, Any]] = []
    for b, st in sorted(block_stats.items()):
        conf = st.get("conf") or []
        block_rows.append({
            "block_idx": b,
            "selected_tokens_total": st.get("selected_tokens_total", 0),
            "local_rounds": st.get("local_rounds", 0),
            "fallback_steps": st.get("fallback_steps", 0),
            "mean_confidence": (sum(conf) / len(conf)) if conf else "",
        })

    generation_chain_rows, chain_block_stats, final_visible_by_block = _build_generation_chain_rows(
        rows,
        gen_length=gen_length,
        block_len=block_len,
        num_blocks=num_blocks,
    )

    metrics = {
        "model": getattr(result, "model", None),
        "benchmark": condition.get("benchmark"),
        "sample_idx": sample.get("sample_id"),
        "sample_id": sample.get("sample_id"),
        "condition_key": condition.get("condition"),
        "experiment": condition.get("experiment"),
        "gen_length": gen_length,
        "gen_steps": int(params.get("gen_steps", params.get("w1_steps", params.get("steps", gen_length or 0))) or 0),
        "gen_blocksize": block_len,
        "block_length": block_len,
        "threshold_label": _threshold_label(params.get("token_selection_confidence_threshold")),
        "token_selection_confidence_threshold": params.get("token_selection_confidence_threshold"),
        "completion_rate": result.trace.get("completion_rate") if result.trace else None,
        "actual_parallelism": result.trace.get("actual_parallelism") if result.trace else None,
        "actual_arness": result.trace.get("actual_arness") if result.trace else None,
        "threshold_pass_rate": result.trace.get("threshold_pass_rate") if result.trace else None,
        "fallback_rate": result.trace.get("fallback_rate") if result.trace else None,
        "prediction": result.text,
        "answer": sample.get("answer"),
        "elapsed_seconds": result.elapsed,
        "tokens_generated": result.tokens_generated,
        "trace_step_rows": len(rows),
        "generation_chain_rows": len(generation_chain_rows),
        "block_stats": chain_block_stats,
        "sample_dir": str(sample_dir),
        "run_dir": str(out_dir),
    }
    metrics.update(final_visible_by_block)
    gen_steps_value = _to_float(metrics.get("gen_steps"), None)
    if gen_steps_value and gen_steps_value != 0:
        metrics["planned_parallelism"] = gen_length / gen_steps_value if gen_length else None
        metrics["nominal_arness"] = gen_steps_value / gen_length if gen_length else None
    else:
        metrics["planned_parallelism"] = None
        metrics["nominal_arness"] = None

    sanitized_step_rows = _sanitize_mask_obj(step_rows)
    sanitized_timeline_rows = _sanitize_mask_obj(timeline_rows)
    sanitized_block_rows = _sanitize_mask_obj(block_rows)
    sanitized_generation_chain_rows = _sanitize_mask_obj(generation_chain_rows)
    sanitized_metrics = _sanitize_mask_obj(metrics)

    write_csv(sample_dir / "generation_chain.csv", sanitized_generation_chain_rows)
    write_csv(sample_dir / "step_events.csv", sanitized_step_rows)
    write_csv(sample_dir / "block_timeline.csv", sanitized_timeline_rows)
    write_csv(sample_dir / "block_metrics.csv", sanitized_block_rows)
    write_json(sample_dir / "metrics.json", sanitized_metrics)
    write_json(sample_dir / "sample_metrics.json", sanitized_metrics)
    (sample_dir / "final_prediction.txt").write_text(_expand_mask_runs(result.text or ""), encoding="utf-8")
    (sample_dir / "problem_groundtruth_prediction.txt").write_text(
        "PROMPT:\n" + str(sample.get("prompt", "")) + "\n\nGROUND TRUTH:\n" + str(sample.get("answer", "")) + "\n\nPREDICTION:\n" + _expand_mask_runs(result.text or ""),
        encoding="utf-8",
    )

    flat_metrics = dict(sanitized_metrics)
    flat_metrics.pop("block_stats", None)
    _write_trace_metric_indexes(out_dir / "sample_traces", flat_metrics)


def write_trace_artifacts(out_dir: Path, adapter: BaseAdapter, condition: Dict[str, Any], sample: Dict[str, Any], result: GenerationResult) -> None:
    trace = result.trace
    if trace is None:
        return
    trace_path = out_dir / "trace.jsonl"
    rows: List[Dict[str, Any]] = []
    tokenizer = getattr(adapter, "tokenizer", None)
    mask_id = int(condition.get("params", {}).get("mask_id", getattr(adapter, "mask_id", -999999)) or -999999)
    for step in trace.get("step_stats") or []:
        raw_selected_ids = _as_int_list(step.get("selected_token_ids") or [])
        raw_positions = _as_int_list(step.get("selected_positions") or [])
        # Keep selected_positions aligned with non-mask selected_token_ids.  W1
        # trace rows may contain separate mask_positions for revisions/remasks.
        selected_pairs = [(p, t) for p, t in zip(raw_positions, raw_selected_ids) if t != mask_id]
        if selected_pairs:
            positions = [p for p, _ in selected_pairs]
            selected_ids = [t for _, t in selected_pairs]
        else:
            positions = raw_positions if not raw_selected_ids else []
            selected_ids = [t for t in raw_selected_ids if t != mask_id]
        decoded = step.get("selected_decoded_tokens")
        if not decoded or len(decoded) != len(selected_ids):
            decoded = _decode_ids(tokenizer, selected_ids, mask_id=mask_id)
        row = {
            "task_id": condition["task"],
            "sample_id": sample.get("sample_id"),
            "sample_idx": sample.get("sample_id"),
            "benchmark": condition["benchmark"],
            "decoding_config_name": condition["condition"],
            "step_idx": step.get("step_idx"),
            "block_idx": step.get("block_idx"),
            "mask_count_before": step.get("mask_count_before"),
            "mask_count_after": step.get("mask_count_after"),
            "selected_count": len(positions),
            "selected_positions": positions,
            "selected_token_ids": selected_ids,
            "selected_decoded_tokens": _sanitize_mask_obj(decoded),
            "selected_confidences": step.get("selected_confidences") or [],
            "changed_positions": step.get("changed_positions") or positions,
            "changed_count": step.get("changed_count", step.get("w1_selected_count", len(positions))),
            "mask_positions": step.get("mask_positions") or [],
            "mask_update_count": step.get("mask_update_count", 0),
            "visible_positions": step.get("visible_positions") or [],
            "visible_count": step.get("visible_count"),
            "state_boxes": _compact_boxes(step.get("state_boxes") or step.get("state_boxes_compact") or ""),
            "mask_boxes": step.get("mask_boxes") or _compact_boxes(_mask_boxes(step.get("mask_count_after", 0))),
            "candidate_positions": step.get("candidate_positions"),
            "candidate_confidences": step.get("candidate_confidences"),
            "transfer_reason": step.get("transfer_reason"),
            "cumulative_transferred_tokens": step.get("cumulative_transferred_tokens"),
            "current_completion_rate": step.get("current_completion_rate"),
            "w1_selected_count": step.get("w1_selected_count"),
            "w1_metadata": step.get("w1_metadata"),
        }
        rows.append(row)
    rows = _sanitize_mask_obj(rows)
    for row in rows:
        append_jsonl(trace_path, row)

    sample_dir = out_dir / "sample_traces" / f"sample_{int(sample.get('sample_id', 0)):04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_json(sample_dir / "trace.json", _sanitize_mask_obj(trace))
    write_json(sample_dir / "sample.json", _sanitize_mask_obj({"sample": sample, "prediction": result.text, "condition": condition}))
    (sample_dir / "prediction.txt").write_text(_expand_mask_runs(result.text), encoding="utf-8")
    _export_canonical_sample_trace(out_dir, sample_dir, condition, sample, result, rows)

def run_condition(adapter: BaseAdapter, model_cfg: Dict[str, Any], condition: Dict[str, Any], inputs: List[Dict[str, Any]], out_dir: Path, force: bool) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = out_dir / "outputs.jsonl"
    summary_path = out_dir / "summary.jsonl"
    if force:
        for path in [outputs_path, summary_path, out_dir / "trace.jsonl"]:
            if path.exists():
                path.unlink()
        sample_traces = out_dir / "sample_traces"
        if sample_traces.exists():
            shutil.rmtree(sample_traces)
    elif outputs_path.exists():
        existing = read_jsonl(outputs_path)
        if len(existing) >= len(inputs):
            return {"status": "skipped_existing", "num_samples": len(existing)}

    shutil.copyfile(prepared_source_path(condition), out_dir / "inputs.jsonl") if prepared_source_path(condition).exists() else write_jsonl(out_dir / "inputs.jsonl", inputs)
    write_json(out_dir / "run.json", {
        "created_at": utc_now(),
        "model": model_cfg,
        "model_alias": adapter.alias,
        "task": condition["task"],
        "experiment": condition["experiment"],
        "benchmark": condition["benchmark"],
        "condition": condition["condition"],
        "params": condition["params"],
        "num_samples": len(inputs),
    })

    telemetry = None
    if model_cfg.get("gpu_telemetry", True):
        telemetry = GpuTelemetry(out_dir / "gpu.csv", interval=float(model_cfg.get("gpu_telemetry_interval", 5)))
        telemetry.start()

    started_all = time.perf_counter()
    try:
        for local_idx, sample in enumerate(inputs):
            prompt = str(sample["prompt"])
            started = time.perf_counter()
            try:
                result = adapter.generate_one(prompt, condition["params"])
                error = None
            except Exception as exc:
                elapsed = time.perf_counter() - started
                result = GenerationResult(text="", elapsed=elapsed, tokens_generated=None, trace=None)
                error = repr(exc)

            tokens = result.tokens_generated
            tps = (tokens / result.elapsed) if tokens and result.elapsed > 0 else None
            out_row = {
                "model": adapter.alias,
                "task": condition["task"],
                "experiment": condition["experiment"],
                "benchmark": condition["benchmark"],
                "condition": condition["condition"],
                "sample_id": sample.get("sample_id", local_idx),
                "dataset_index": sample.get("dataset_index"),
                "prompt": prompt,
                "answer": sample.get("answer"),
                "raw_output": result.text,
                "prediction": result.text,
                "params": condition["params"],
                "metadata": sample.get("metadata", {}),
                "timing": {
                    "elapsed_seconds": round(result.elapsed, 6),
                    "tokens_generated": tokens,
                    "tokens_per_second": round(tps, 6) if tps is not None else None,
                },
                "error": error,
            }
            append_jsonl(outputs_path, out_row)

            trace = result.trace
            summary_row = {
                "model": adapter.alias,
                "benchmark": condition["benchmark"],
                "sample_idx": sample.get("sample_id", local_idx),
                "sample_id": sample.get("sample_id", local_idx),
                "decoding_config_name": condition["condition"],
                "input": prompt,
                "prediction": result.text,
                "elapsed_seconds": round(result.elapsed, 6),
                "tokens_per_second": round(tps, 6) if tps is not None else None,
                "steps": int(condition["params"].get("gen_steps", condition["params"].get("gen_length", 128))),
                "gen_length": int(condition["params"].get("gen_length", 128)),
                "block_length": int(condition["params"].get("gen_blocksize", condition["params"].get("gen_length", 128))),
                "completion_rate": trace.get("completion_rate") if trace else None,
                "actual_parallelism": trace.get("actual_parallelism") if trace else None,
                "actual_arness": trace.get("actual_arness") if trace else None,
                "threshold_pass_rate": trace.get("threshold_pass_rate") if trace else None,
                "fallback_rate": trace.get("fallback_rate") if trace else None,
                "error": error,
            }
            append_jsonl(summary_path, summary_row)
            write_trace_artifacts(out_dir, adapter, condition, sample, result)
    finally:
        if telemetry is not None:
            telemetry.stop()

    elapsed_all = time.perf_counter() - started_all
    manifest = {"status": "finished", "num_samples": len(inputs), "elapsed_seconds": round(elapsed_all, 3)}
    write_json(out_dir / "output_manifest.json", manifest)
    if condition["params"].get("return_trace"):
        sample_idx = inputs[0].get("sample_id", 0) if inputs else 0
        command = f'python visual_arness_trace.py "{out_dir}" --sample-idx {sample_idx}\n'
        (out_dir / "visual_command.txt").write_text(command, encoding="utf-8")
    return manifest


# Small mutable bridge used by run_condition for exact input copy without making
# its signature noisier. It is set just before calling run_condition.
_PREPARED_SOURCE: Optional[Path] = None


def prepared_source_path(condition: Dict[str, Any]) -> Path:
    return _PREPARED_SOURCE or Path("__missing__")


def maybe_auto_prepare(config_path: Path, only: Sequence[str], force: bool = False) -> None:
    import prepare_data
    argv = ["--config", str(config_path)]
    if only:
        argv += ["--only", *only]
    if force:
        argv.append("--force")
    old_argv = sys.argv
    try:
        sys.argv = ["prepare_data.py"] + argv
        prepare_data.main()
    finally:
        sys.argv = old_argv


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate model outputs natively without OpenCompass inference.")
    parser.add_argument("--config", default="test_config.yaml")
    parser.add_argument("--only", nargs="*", default=[], help="Task or experiment names to run.")
    parser.add_argument("--models", nargs="*", default=None, help="Model aliases to run. Defaults to experiment models or all config models.")
    parser.add_argument("--output-root", default="model_outputs")
    parser.add_argument("--prepared-root", default=None, help="Override prepared root. Default: data.prepared_dir.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auto-prepare", action="store_true", help="Run prepare_data.py first if inputs are missing.")
    parser.add_argument(
        "--write-global-manifest",
        action="store_true",
        help=(
            "Also write aggregate manifests under <output-root>/_manifests/. "
            "Default: only per-model manifests under <output-root>/<model>/ are written."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    selected = set(args.only or [])

    all_models = normalize_models(config)
    explicit_models = set(args.models or []) if args.models else None
    conditions = list(iter_conditions(config, selected))
    if not conditions:
        raise SystemExit("No experiments matched.")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    if args.auto_prepare:
        maybe_auto_prepare(config_path, args.only, force=False)

    plan: List[Tuple[Dict[str, Any], Dict[str, Any], Path, Path]] = []
    for condition in conditions:
        if args.prepared_root is None:
            input_path = prepared_input_path(config, condition)
        else:
            base_override = Path(args.prepared_root)
            if not base_override.is_absolute():
                base_override = ROOT / base_override
            if str(condition["benchmark"]) in {"ruler_niah_single_1", "ruler_niah_order_2", "ruler_niah_double_2"}:
                input_path = prepared_file_for(base_override, condition["benchmark"], condition["params"])
            else:
                input_path = base_override / str(condition["benchmark"]) / "original_source"
        for model_cfg in all_models:
            alias = model_alias(model_cfg)
            if not model_selected(model_cfg, explicit_models):
                continue
            exp_models = condition.get("experiment_models")
            if not model_selected(model_cfg, exp_models):
                continue
            out_dir = output_dir_for(output_root, alias, condition)
            plan.append((model_cfg, condition, input_path, out_dir))

    if not plan:
        raise SystemExit("No model/condition pairs selected.")

    print(f"Native generation plan: {len(plan)} run(s), {len(set(model_alias(p[0]) for p in plan))} model(s).")
    for model_cfg, condition, input_path, out_dir in plan:
        print(f"- {model_alias(model_cfg)} | {condition['experiment']} | {condition['condition']} | inputs={input_path} | out={out_dir}")
    if args.dry_run:
        return 0

    # Load once per model and run all selected conditions for that model.
    by_model: Dict[str, List[Tuple[Dict[str, Any], Path, Path]]] = {}
    model_cfg_by_alias: Dict[str, Dict[str, Any]] = {}
    for model_cfg, condition, input_path, out_dir in plan:
        alias = model_alias(model_cfg)
        by_model.setdefault(alias, []).append((condition, input_path, out_dir))
        model_cfg_by_alias[alias] = model_cfg

    global _PREPARED_SOURCE
    manifest_rows: List[Dict[str, Any]] = []
    for alias, jobs in by_model.items():
        model_cfg = model_cfg_by_alias[alias]
        print(f"\n[LOAD] {alias} backend={model_cfg.get('backend')} path={model_cfg.get('path') or model_cfg.get('model')}", flush=True)
        adapter = build_adapter(model_cfg)
        try:
            for condition, input_path, out_dir in jobs:
                inputs, source_path, source_label = load_existing_or_prepared_inputs(config, condition, input_path)
                _PREPARED_SOURCE = source_path
                print(f"[RUN] {alias} | {condition['experiment']} | {condition['condition']} | n={len(inputs)} | inputs={source_label}", flush=True)
                result = run_condition(adapter, model_cfg, condition, inputs, out_dir, force=args.force)
                manifest_rows.append({
                    "model": alias,
                    "task": condition["task"],
                    "experiment": condition["experiment"],
                    "benchmark": condition["benchmark"],
                    "condition": condition["condition"],
                    "output_dir": str(out_dir),
                    "input_path": source_label,
                    **result,
                })
        finally:
            adapter.close()

    manifest_paths = write_model_manifests(
        output_root,
        "native_model_manifest",
        manifest_rows,
        write_global=args.write_global_manifest,
    )
    print("\nWrote per-model generation manifests:")
    for path in manifest_paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
