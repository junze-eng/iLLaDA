#!/usr/bin/env python3
"""Native GPU generation runner for iLLaDA / W1 experiments.

This replaces the OpenCompass inference path for mechanism-oriented tests.  It
reads the same `test_config.yaml`, reuses prepared `inputs.jsonl` files produced
by `prepare_data.py`, and writes model outputs under:

    model_outputs/<model_alias>/<task>/<benchmark_alias>/<condition>/

Important efficiency property: the model is loaded once per selected model, then
all selected sweep conditions are generated in the same process.  This avoids the
old repeated load cost across gen_steps/threshold/config sweeps.

Supported backends:
  - illada: native masked-diffusion generation via generate.py
  - hf_causal / hf: HuggingFace AutoModelForCausalLM.generate
  - openai_compatible / api: POST /chat/completions style local/API model

If the config has no top-level `models:` block, the legacy top-level `model:`
block is treated as an iLLaDA model named by `abbr` or `illada`.
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

from prepare_data import bench_alias, condition_name, prepared_dir_for, read_jsonl, write_jsonl, json_default


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


def model_alias(model_cfg: Dict[str, Any]) -> str:
    return safe_name(str(model_cfg.get("name") or model_cfg.get("abbr") or Path(str(model_cfg.get("path", "model"))).name or "model"))


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
        legacy.setdefault("name", legacy.get("abbr", "illada"))
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
    return root / safe_name(model_name) / safe_name(condition["task"] or "runs") / bench_alias(condition["benchmark"]) / condition["condition"]


def prepared_input_path(config: Dict[str, Any], condition: Dict[str, Any]) -> Path:
    data_cfg = config.get("data", {}) or {}
    base = Path(data_cfg.get("prepared_dir", "data/prepared")) / "native"
    if not base.is_absolute():
        base = ROOT / base
    return prepared_dir_for(base, condition["task"], condition["benchmark"], condition["params"]) / "inputs.jsonl"


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
    if backend in {"openai_compatible", "api", "w1", "vllm"}:
        return OpenAICompatibleAdapter(cfg)
    raise SystemExit(f"Unsupported model backend `{backend}` for model {cfg.get('name')}")


def write_trace_artifacts(out_dir: Path, adapter: BaseAdapter, condition: Dict[str, Any], sample: Dict[str, Any], result: GenerationResult) -> None:
    trace = result.trace
    if trace is None:
        return
    trace_path = out_dir / "trace.jsonl"
    rows = []
    tokenizer = getattr(adapter, "tokenizer", None)
    for step in trace.get("step_stats") or []:
        selected_ids = step.get("selected_token_ids") or []
        decoded = []
        if tokenizer is not None:
            for token_id in selected_ids:
                try:
                    decoded.append(tokenizer.decode([int(token_id)], skip_special_tokens=False))
                except Exception:
                    decoded.append(str(token_id))
        rows.append({
            "task_id": condition["task"],
            "sample_id": sample.get("sample_id"),
            "sample_idx": sample.get("sample_id"),
            "benchmark": condition["benchmark"],
            "decoding_config_name": condition["condition"],
            "step_idx": step.get("step_idx"),
            "block_idx": step.get("block_idx"),
            "mask_count_before": step.get("mask_count_before"),
            "mask_count_after": step.get("mask_count_after"),
            "selected_positions": step.get("selected_positions") or [],
            "selected_token_ids": selected_ids,
            "selected_decoded_tokens": decoded,
            "selected_confidences": step.get("selected_confidences") or [],
            "candidate_positions": step.get("candidate_positions"),
            "candidate_confidences": step.get("candidate_confidences"),
            "transfer_reason": step.get("transfer_reason"),
            "cumulative_transferred_tokens": step.get("cumulative_transferred_tokens"),
            "current_completion_rate": step.get("current_completion_rate"),
        })
    for row in rows:
        append_jsonl(trace_path, row)

    sample_dir = out_dir / "sample_traces" / f"sample_{int(sample.get('sample_id', 0)):04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_json(sample_dir / "trace.json", trace)
    write_json(sample_dir / "sample.json", {"sample": sample, "prediction": result.text, "condition": condition})
    (sample_dir / "prediction.txt").write_text(result.text, encoding="utf-8")


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
    parser.add_argument("--prepared-root", default=None, help="Override prepared root. Default: data.prepared_dir/native.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auto-prepare", action="store_true", help="Run prepare_data.py first if inputs are missing.")
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
        input_path = prepared_input_path(config, condition) if args.prepared_root is None else prepared_dir_for(Path(args.prepared_root), condition["task"], condition["benchmark"], condition["params"]) / "inputs.jsonl"
        if not input_path.exists() and not args.dry_run:
            raise SystemExit(
                f"Missing prepared inputs: {input_path}\n"
                f"Run: python prepare_data.py --config {config_path} --only {condition['experiment']}"
            )
        for model_cfg in all_models:
            alias = model_alias(model_cfg)
            if explicit_models and alias not in explicit_models and str(model_cfg.get("name")) not in explicit_models:
                continue
            exp_models = condition.get("experiment_models")
            if exp_models and alias not in exp_models and str(model_cfg.get("name")) not in exp_models:
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
                inputs = read_jsonl(input_path)
                _PREPARED_SOURCE = input_path
                print(f"[RUN] {alias} | {condition['experiment']} | {condition['condition']} | n={len(inputs)}", flush=True)
                result = run_condition(adapter, model_cfg, condition, inputs, out_dir, force=args.force)
                manifest_rows.append({
                    "model": alias,
                    "task": condition["task"],
                    "experiment": condition["experiment"],
                    "benchmark": condition["benchmark"],
                    "condition": condition["condition"],
                    "output_dir": str(out_dir),
                    "input_path": str(input_path),
                    **result,
                })
        finally:
            adapter.close()

    write_jsonl(output_root / "native_output_manifest.jsonl", manifest_rows)
    write_csv(output_root / "native_output_manifest.csv", manifest_rows)
    print(f"\nWrote manifest: {output_root / 'native_output_manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
