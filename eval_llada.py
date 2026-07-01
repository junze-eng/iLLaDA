'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import json
import torch
import re
import time
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from generate import generate


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        token_selection_confidence_threshold=None,
        min_transfer_tokens=1,
        metrics_output=None,
        per_sample_output=None,
        step_trace_output=None,
        return_trace=False,
        trace_token_snapshots=False,
        trace_decode_snapshots=False,
        device="cuda",
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()
        self.model_path = model_path

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        self._rank = 0
        self._world_size = 1
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.token_selection_confidence_threshold = token_selection_confidence_threshold
        self.min_transfer_tokens = int(min_transfer_tokens)
        self.per_sample_output = per_sample_output or metrics_output
        self.step_trace_output = step_trace_output
        self.metrics_output = self.per_sample_output
        self.return_trace = return_trace
        self.trace_token_snapshots = bool(trace_token_snapshots)
        self.trace_decode_snapshots = bool(trace_decode_snapshots)

        if self.rank == 0:
            for output_path in (self.per_sample_output, self.step_trace_output):
                if output_path:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def _cuda_stats_before(self):
        if self.device.type != 'cuda':
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

    def _cuda_stats_after(self):
        if self.device.type != 'cuda':
            return {}
        torch.cuda.synchronize(self.device)
        return {
            'cuda_max_memory_allocated_mb': round(torch.cuda.max_memory_allocated(self.device) / 1024 ** 2, 3),
            'cuda_max_memory_reserved_mb': round(torch.cuda.max_memory_reserved(self.device) / 1024 ** 2, 3),
        }

    def _write_jsonl(self, output_path, records):
        if not output_path or self.rank != 0:
            return
        with open(output_path, 'a', encoding='utf-8') as f:
            for record in records:
                record = dict(record)
                record['rank'] = self.rank
                record['world_size'] = self.world_size
                record['model_path'] = getattr(self, 'model_path', None)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _trace_with_decoded_snapshots(self, trace):
        if not trace or not self.trace_decode_snapshots:
            return trace
        trace = dict(trace)
        snapshots = []
        for snapshot in trace.get('token_snapshots') or []:
            item = dict(snapshot)
            decoded = []
            for token_ids in item.get('generated_token_ids') or []:
                decoded.append(self.tokenizer.decode(token_ids, skip_special_tokens=False))
            item['generated_text'] = decoded
            snapshots.append(item)
        trace['token_snapshots'] = snapshots
        return trace

    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        for sample_idx, elem in enumerate(tqdm(ds, desc="Generating...")):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]

            self._cuda_stats_before()
            profile_trace = self.return_trace or bool(self.per_sample_output)
            started = time.perf_counter()
            generated = generate(
                self.model,
                prompt,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0,
                cfg_scale=self.cfg,
                remasking=self.remasking,
                mask_id=self.mask_id,
                token_selection_confidence_threshold=self.token_selection_confidence_threshold,
                min_transfer_tokens=self.min_transfer_tokens,
                return_trace=profile_trace,
                trace_token_snapshots=self.trace_token_snapshots or self.trace_decode_snapshots,
                tokenizer=self.tokenizer,
            )
            elapsed = time.perf_counter() - started
            cuda_stats = self._cuda_stats_after()
            if profile_trace:
                generated_answer, trace = generated
            else:
                generated_answer, trace = generated, None
            
            generated_answer = self.tokenizer.decode(generated_answer[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)

            generated_tokens = self.gen_length
            per_sample_record = {
                'sample_idx': sample_idx,
                'input': elem["question_text"],
                'prediction': generated_answer,
                'correctness': None,
                'score': None,
                'evaluator_status': 'pending_lm_eval_aggregation',
                'prompt_tokens': int(prompt.shape[1]),
                'generated_tokens': int(generated_tokens),
                'elapsed_seconds': round(elapsed, 6),
                'tokens_per_second': round(generated_tokens / elapsed, 6) if elapsed > 0 else None,
                'steps': int(self.steps),
                'gen_length': int(self.gen_length),
                'block_length': int(self.block_length),
                'mask_id': int(self.mask_id),
                'remasking': self.remasking,
                'cfg': float(self.cfg),
                'token_selection_confidence_threshold': self.token_selection_confidence_threshold,
                'min_transfer_tokens': self.min_transfer_tokens,
                'forward_passes': int(self.steps),
                'effective_parallelism': float(self.gen_length / self.steps) if self.steps else None,
                'planned_parallelism': float(self.gen_length / self.steps) if self.steps else None,
                'arness': float(self.steps / self.gen_length) if self.gen_length else None,
                'final_mask_count': trace.get('final_mask_count') if trace else None,
                'remaining_masks': trace.get('final_mask_count') if trace else None,
                'completion_rate': trace.get('completion_rate') if trace else None,
                'actual_parallelism': trace.get('actual_parallelism') if trace else None,
                'scheduled_transfer_count': trace.get('scheduled_transfer_count') if trace else None,
                'threshold_passed_count': trace.get('threshold_passed_count') if trace else None,
                'fallback_forced_count': trace.get('fallback_forced_count') if trace else None,
                'actual_transfer_count': trace.get('actual_transfer_count') if trace else None,
                **cuda_stats,
            }
            self._write_jsonl(self.per_sample_output, [per_sample_record])
            if trace is not None:
                self._write_jsonl(self.step_trace_output, [{
                    'sample_idx': sample_idx,
                    'trace': self._trace_with_decoded_snapshots(trace),
                }])

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
    
