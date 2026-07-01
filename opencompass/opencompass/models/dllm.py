import os
import sys
import json
import time
from pathlib import Path
llada_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(llada_root))
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
##use llada generate
from generate import generate as LLaDA_generate
import torch.nn.functional as F
import numpy as np
PromptType = Union[PromptList, str]
def _get_meta_template(meta_template):
    default_meta_template = dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            # XXX: all system roles are mapped to human in purpose
            # dict(role='SYSTEM', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ],
        reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')]
    )
    return APITemplateParser(meta_template or default_meta_template)

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@MODELS.register_module()
class LLaDAModel(BaseModel):
    """Model wrapper around LLaDA model.

    Args:
        path (str): The name or path to LLaDA model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        extract_pred_after_decode (bool): Whether to extract the prediction
            string from the decoded output string, instead of extract the
            prediction tokens before decoding. Defaults to False.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.
        pad_token_id (int): The id of the padding token. Defaults to None. Use
            (#vocab + pad_token_id) if get negative value.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
        use_fastchat_template (str, optional): Whether to use fastchat to get
            the conversation template. If True, fastchat needs to be
            implemented first. Defaults to False.
        end_str (str, optional): Whether to trim generated strings with end_str
            if the model has special ending strings that are not handled well.
            Defaults to None.

    Note:
        About ``extract_pred_after_decode``: Commonly, we should extract the
        the prediction tokens before decoding. But for some tokenizers using
        ``sentencepiece``, like LLaMA,  this behavior may change the number of
        whitespaces, which is harmful for Python programming tasks.
    """

    def __init__(self,
                 path: str,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 use_fastchat_template: bool = False,
                 end_str: Optional[str] = None,
                 stop_words: Optional[List[str]] = [],
                 cfg = 0,
                 temperature = 0.,
                 remasking = 'low_confidence', # 'random'
                 mask_id = 126336, # The token id of [MASK] is 126336.
                 padding_id = 126081, # The token id of <pad> is 126081.
                 mc_num = 1,
                 gen_steps = 512,
                 gen_length = 512,
                 gen_blocksize = 512,
                 batch_size_ = 1,
                 diff_confidence_eos_eot_inf = False,
                 diff_logits_eos_inf = False,
                 token_selection_confidence_threshold = None,
                 min_transfer_tokens = 1,
                 metrics_output = None,
                 per_sample_output = None,
                 step_trace_output = None,
                 return_trace = False,
                 trace_token_snapshots = False,
                 trace_decode_snapshots = False,
                 context_prefix_tokens = 0,
                 context_prefix_text = 'Context padding sentence. ',
                 benchmark = None,
                 task_id = None,
                 decoding_config_name = None,
                 context_length = None,
                 needle_position = None,
                 **kwargs,
                 ) -> None:
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template)
        if hf_cache_dir is None:
            hf_cache_dir = os.getenv('HF_MODEL_HUB', None)
        self.logger = get_logger()
        self.pad_token_id = pad_token_id
        assert mode in ['none', 'mid']
        self.mode = mode
        self._load_tokenizer(path=path,
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(path=path,
                             model_kwargs=model_kwargs,
                             peft_path=peft_path)
        self.generation_kwargs = generation_kwargs
        self.use_fastchat_template = use_fastchat_template
        self.end_str = end_str
        self.stop_words = stop_words
        ## add cfg and mc_num
        self.cfg = cfg
        self.mc_num = mc_num

        ##Todo: modify input
        self.batch_size = batch_size_
        self.gen_steps = gen_steps
        self.gen_length = gen_length
        self.gen_blocksize = gen_blocksize
        self.temperature = temperature
        self.remasking = remasking
        self.padding_id = padding_id
        self.mask_id = mask_id
        self.diff_confidence_eos_eot_inf = diff_confidence_eos_eot_inf
        self.diff_logits_eos_inf = diff_logits_eos_inf
        self.token_selection_confidence_threshold = token_selection_confidence_threshold
        self.min_transfer_tokens = int(min_transfer_tokens)
        self.per_sample_output = per_sample_output or metrics_output
        self.step_trace_output = step_trace_output
        self.metrics_output = self.per_sample_output
        self.return_trace = return_trace
        self.trace_token_snapshots = bool(trace_token_snapshots)
        self.trace_decode_snapshots = bool(trace_decode_snapshots)
        self.context_prefix_tokens = int(context_prefix_tokens)
        self.context_prefix_text = context_prefix_text
        self.benchmark = benchmark
        self.task_id = task_id
        self.decoding_config_name = decoding_config_name
        self.context_length = context_length
        self.needle_position = needle_position
        self._profile_sample_idx = 0
        for output_path in (self.per_sample_output, self.step_trace_output):
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        self.template_parser = _get_meta_template(meta_template)

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):
        from transformers import AutoTokenizer
        tokenizer_kwargs = dict(tokenizer_kwargs)
        tokenizer_kwargs.setdefault('trust_remote_code', True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path, **tokenizer_kwargs)

        # A patch for some models without pad_token_id
        if self.pad_token_id is not None:
            if self.pad_token_id < 0:
                self.pad_token_id += self.tokenizer.vocab_size
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f'Using {self.pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != self.pad_token_id:
                self.logger.warning(
                    'pad_token_id is not consistent with the tokenizer. Using '
                    f'{self.pad_token_id} as pad_token_id')
            self.tokenizer.pad_token_id = self.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            self.logger.warning('pad_token_id is not set for the tokenizer.')
            if self.tokenizer.eos_token is not None:
                self.logger.warning(
                    f'Using eos_token_id {self.tokenizer.eos_token} '
                    'as pad_token_id.')
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                from transformers.generation import GenerationConfig
                gcfg = GenerationConfig.from_pretrained(path)

                if gcfg.pad_token_id is not None:
                    self.logger.warning(
                        f'Using pad_token_id {gcfg.pad_token_id} '
                        'as pad_token_id.')
                    self.tokenizer.pad_token_id = gcfg.pad_token_id
                else:
                    raise ValueError(
                        'pad_token_id is not set for this tokenizer. Try to '
                        'set pad_token_id via passing '
                        '`pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path or \
                (tokenizer_path and
                 'decapoda-research/llama' in tokenizer_path):
            self.logger.warning('We set new pad_token_id for LLaMA model')
            # keep consistent with official LLaMA repo
            # https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb  # noqa
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.pad_token_id = 0

    def _set_model_kwargs_torch_dtype(self, model_kwargs):
        if 'torch_dtype' not in model_kwargs:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = {
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'torch.float': torch.float,
                'auto': 'auto',
                'None': None
            }.get(model_kwargs['torch_dtype'])
        self.logger.debug(f'HF using torch_dtype: {torch_dtype}')
        if torch_dtype is not None:
            model_kwargs['torch_dtype'] = torch_dtype

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        from transformers import AutoModel, AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, trust_remote_code = True,**model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(
                path, trust_remote_code=True, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
        if self.model.generation_config is not None:
            self.model.generation_config.do_sample = False

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path:
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2
            self.model.config.pad_token_id = self.tokenizer.pad_token_id


    def _get_loglikelihood(self, inputs: str, conts: str) -> float:
        """Get loglikelihood scores given input string and continuation string.

        Args:
            inputs (str): string.
            conts (str): strings: slices after the space.
        Returns:
            float: loglikelihood scores.
        """
        input_tokenizer_out = self.tokenizer(inputs,
                                             padding=True,
                                             truncation=False,
                                             return_length=True,
                                             return_tensors='pt').to(
                                                 self.model.device)

        input_ids = input_tokenizer_out['input_ids'][:, :self.max_seq_len]
        input_length = input_tokenizer_out['length']
        context_ids = [
            self.tokenizer(inputs[i].replace(conts[i], ''),
                           padding=False,
                           truncation=True,
                           max_length=self.max_seq_len)['input_ids']
            for i in range(len(inputs))
        ]
        # forward
        outputs = self.model(input_ids)['logits']
        outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
        # calculate loglikelihood
        answer = np.zeros(len(inputs))
        for i in range(len(inputs)):
            if self.tokenizer.padding_side == 'right':
                cont_ids = input_ids[i, len(context_ids[i]):input_length[i]]
                logits = outputs[i,
                                 len(context_ids[i]) - 1:input_length[i] -
                                 1, :]  # noqa
            else:
                cont_ids = input_ids[i, len(context_ids[i]) - input_length[i]:]
                logits = outputs[i,
                                 len(context_ids[i]) - input_length[i] - 1:-1]
            # Reducing the dimension will lead to a wrong outcome
            logits_gather = torch.gather(
                logits.unsqueeze(0), 2,
                cont_ids.unsqueeze(0).unsqueeze(-1))  # [1, seq]
            # Answer: sum the likelihood of each token in continuation
            answer[i] = float(logits_gather.detach().cpu().sum())
        return answer

    def get_mink_percent(self, inputs: List[str], k: int = 20) -> List[float]:
        """https://swj0419.github.io/detect-pretrain.github.io/"""

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_mink_percent(inputs, k=k)
        else:
            return np.concatenate([
                self._get_mink_percent(inputs=[text], k=k) for text in inputs
            ])

    def _get_mink_percent(self, inputs: List[str], k: int = 20) -> List[float]:
        outputs, inputs = self.get_logits(inputs)
        shift_logits = outputs[:, :-1, :].contiguous().float()
        shift_labels = inputs['tokens']['input_ids'][:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())
        lens = (inputs['tokens']['input_ids'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        mink_percent = []
        for nloss, nlen in zip(loss, lens):
            nlen = int(nlen)
            minklen = max(nlen * k // 100, 1)
            nloss = torch.topk(loss[-nlen:], minklen, dim=-1)[0]
            nloss = -nloss.float().mean().cpu().detach().numpy()
            mink_percent.append(nloss)
        return np.array(mink_percent)

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))

    def _cuda_stats_before(self):
        device = getattr(self.model, 'device', None)
        if device is None or device.type != 'cuda':
            return
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    def _cuda_stats_after(self):
        device = getattr(self.model, 'device', None)
        if device is None or device.type != 'cuda':
            return {}
        torch.cuda.synchronize(device)
        return {
            'cuda_max_memory_allocated_mb': round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 3),
            'cuda_max_memory_reserved_mb': round(torch.cuda.max_memory_reserved(device) / 1024 ** 2, 3),
        }

    def _write_jsonl(self, output_path, records):
        if not output_path:
            return
        with open(output_path, 'a', encoding='utf-8') as f:
            for record in records:
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

    def _failure_type(self, prediction, correctness, trace):
        if correctness == 1:
            return 'correct'
        if prediction is None or str(prediction).strip() == '':
            return 'empty_output'
        if trace and (trace.get('final_mask_count', 0) > 0 or (trace.get('completion_rate') is not None and trace.get('completion_rate') < 1)):
            return 'unfinished_generation'
        lower = str(prediction).strip().lower()
        role_like = lower in ('assistant', '<think>', '</think>') or lower.startswith('assistant') or lower.startswith('<think>')
        if role_like:
            return 'format_failure'
        if self.benchmark and 'ruler_niah' in str(self.benchmark):
            return 'retrieval_failure'
        return 'task_failure'

    def _build_profile_records(self, prompt_texts, tokenized_prompts, responses, prompt_tokens, elapsed, trace, cuda_stats):
        batch_size = len(responses)
        total_generated_tokens = self.gen_length * batch_size
        per_sample_records = []
        step_trace_records = []
        trace_actual_transfer_count = trace.get('actual_transfer_count') if trace else None
        trace_visible_tokens = [
            len(self.tokenizer(response, add_special_tokens=False)['input_ids'])
            for response in responses
        ]
        for i in range(batch_size):
            sample_idx = self._profile_sample_idx
            visible_output_tokens = int(trace_visible_tokens[i])
            record = {
                'task_id': self.task_id,
                'benchmark': self.benchmark,
                'sample_idx': sample_idx,
                'decoding_config_name': self.decoding_config_name,
                'batch_size': int(batch_size),
                'batch_item_idx': int(i),
                'input': prompt_texts[i],
                'tokenized_prompt': tokenized_prompts[i] if i < len(tokenized_prompts) else None,
                'prediction': responses[i],
                'target': None,
                'correctness': None,
                'official_score': None,
                'score': None,
                'evaluator_status': 'pending_opencompass_eval',
                'failure_type': self._failure_type(responses[i], None, trace),
                'prompt_tokens': int(prompt_tokens),
                'generated_tokens': int(self.gen_length),
                'batch_generated_tokens': int(total_generated_tokens),
                'elapsed_seconds': round(elapsed, 6),
                'tokens_per_second': round(total_generated_tokens / elapsed, 6) if elapsed > 0 else None,
                'actual_commit_tps': round(trace_actual_transfer_count / elapsed, 6) if elapsed > 0 and trace_actual_transfer_count is not None else None,
                'visible_tps': round(sum(trace_visible_tokens) / elapsed, 6) if elapsed > 0 else None,
                'visible_output_tokens': visible_output_tokens,
                'steps': int(self.gen_steps),
                'gen_length': int(self.gen_length),
                'block_length': int(self.gen_blocksize),
                'mask_id': int(self.mask_id),
                'remasking': self.remasking,
                'cfg': float(self.cfg),
                'temperature': float(self.temperature),
                'token_selection_confidence_threshold': self.token_selection_confidence_threshold,
                'min_transfer_tokens': self.min_transfer_tokens,
                'context_prefix_tokens': self.context_prefix_tokens,
                'context_length': self.context_length,
                'needle_position': self.needle_position,
                'truncated': False,
                'effective_parallelism': float(self.gen_length / self.gen_steps) if self.gen_steps else None,
                'arness': float(self.gen_steps / self.gen_length) if self.gen_length else None,
                'final_mask_count': trace.get('final_mask_count') if trace else None,
                'completion_rate': trace.get('completion_rate') if trace else None,
                'actual_parallelism': trace.get('actual_parallelism') if trace else None,
                'actual_arness': trace.get('actual_arness') if trace else None,
                'scheduled_transfer_count': trace.get('scheduled_transfer_count') if trace else None,
                'threshold_passed_count': trace.get('threshold_passed_count') if trace else None,
                'fallback_forced_count': trace.get('fallback_forced_count') if trace else None,
                'actual_transfer_count': trace.get('actual_transfer_count') if trace else None,
                'threshold_pass_rate': trace.get('threshold_pass_rate') if trace else None,
                'fallback_rate': trace.get('fallback_rate') if trace else None,
                **cuda_stats,
            }
            per_sample_records.append(record)
            if trace is not None:
                for step in trace.get('step_stats') or []:
                    if int(step.get('batch_item_idx', 0)) != i:
                        continue
                    selected_ids = step.get('selected_token_ids') or []
                    step_trace_records.append({
                        'task_id': self.task_id,
                        'sample_idx': sample_idx,
                        'benchmark': self.benchmark,
                        'decoding_config_name': self.decoding_config_name,
                        'batch_item_idx': int(i),
                        'step_idx': step.get('step_idx'),
                        'block_idx': step.get('block_idx'),
                        'mask_count_before': step.get('mask_count_before'),
                        'mask_count_after': step.get('mask_count_after'),
                        'selected_positions': step.get('selected_positions') or [],
                        'selected_token_ids': selected_ids,
                        'selected_decoded_tokens': [
                            self.tokenizer.decode([token_id], skip_special_tokens=False)
                            for token_id in selected_ids
                        ],
                        'selected_confidences': step.get('selected_confidences') or [],
                        'transfer_reason': step.get('transfer_reason'),
                        'cumulative_transferred_tokens': step.get('cumulative_transferred_tokens'),
                        'current_completion_rate': step.get('current_completion_rate'),
                        'current_partial_output': None,
                        'target_tokens_present': None,
                        'target_tokens_positions': None,
                        'target_tokens_changed_later': None,
                        'output_prefix_stability': None,
                        'syntax_or_format_break_step': None,
                        'final_answer_region_changed_later': None,
                    })
            self._profile_sample_idx += 1
        return per_sample_records, step_trace_records

    def _context_prefix(self) -> str:
        if self.context_prefix_tokens <= 0:
            return ''
        unit = self.context_prefix_text or 'Context padding sentence. '
        repeated = unit
        target = self.context_prefix_tokens
        while len(self.tokenizer.encode(repeated, add_special_tokens=False)) < target:
            repeated += unit
        token_ids = self.tokenizer.encode(repeated, add_special_tokens=False)[:target]
        prefix = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return (
            prefix
            + '\n\nThe preceding context is padding for a context-length stress test. '
            + 'Answer the actual task below.\n\n'
        )

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs. """
        messages = _convert_chat_messages(inputs)
        context_prefix = self._context_prefix()
        if context_prefix:
            for message_group in messages:
                for message in message_group:
                    if message['role'] == 'user':
                        message['content'] = context_prefix + message['content']
                        break
        prompt = [self.tokenizer.apply_chat_template(m_i, add_generation_prompt=True, tokenize=False) for m_i in messages]
        print('steps:', self.gen_steps, 'length:', self.gen_length, 'blocksize:', self.gen_blocksize)
        print('temperature:', self.temperature, 'cfg:', self.cfg, 'remasking:', self.remasking)
        print('mask_id:', self.mask_id, 'padding_id:', self.padding_id)
        print('diff_confidence_eos_eot_inf:', self.diff_confidence_eos_eot_inf, 'diff_logits_eos_inf:', self.diff_logits_eos_inf)
        print('token_selection_confidence_threshold:', self.token_selection_confidence_threshold)
        print('min_transfer_tokens:', self.min_transfer_tokens, 'per_sample_output:', self.per_sample_output, 'step_trace_output:', self.step_trace_output, 'return_trace:', self.return_trace)
        print('trace_token_snapshots:', self.trace_token_snapshots, 'trace_decode_snapshots:', self.trace_decode_snapshots)
        print('final prompt:', prompt)
        prompt_texts = prompt
        self.tokenizer.padding_side = "left" 
        encoded_prompt = self.tokenizer.batch_encode_plus(prompt, padding = True, return_tensors='pt')
        prompt = encoded_prompt['input_ids']
        tokenized_prompts = prompt.detach().cpu().tolist()
        attention_mask = encoded_prompt.get('attention_mask')
        profile_trace = self.return_trace or bool(self.per_sample_output)
        self._cuda_stats_before()
        started = time.perf_counter()
        generated = LLaDA_generate(
            model = self.model,
            prompt = prompt.to(self.model.device),
            attention_mask = attention_mask.to(self.model.device) if attention_mask is not None else None,
            steps = self.gen_steps,
            gen_length = self.gen_length,
            block_length = self.gen_blocksize,
            temperature = self.temperature,
            cfg_scale = self.cfg,
            remasking = self.remasking,
            mask_id = self.mask_id,
            confidence_eos_eot_inf = self.diff_confidence_eos_eot_inf,
            logits_eos_inf = self.diff_logits_eos_inf,
            token_selection_confidence_threshold = self.token_selection_confidence_threshold,
            min_transfer_tokens = self.min_transfer_tokens,
            return_trace = profile_trace,
            trace_token_snapshots = self.trace_token_snapshots or self.trace_decode_snapshots,
            tokenizer = self.tokenizer,
        )
        elapsed = time.perf_counter() - started
        cuda_stats = self._cuda_stats_after()
        if profile_trace:
            x, trace = generated
        else:
            x, trace = generated, None
        responses = []
        batch_size = prompt.shape[0]
        
        for i in range(batch_size):
            responses.append(self.tokenizer.decode(x[i, -self.gen_length:], skip_special_tokens=True))
        per_sample_records, step_trace_records = self._build_profile_records(
            prompt_texts=prompt_texts,
            tokenized_prompts=tokenized_prompts,
            responses=responses,
            prompt_tokens=prompt.shape[1],
            elapsed=elapsed,
            trace=trace,
            cuda_stats=cuda_stats,
        )
        self._write_jsonl(self.per_sample_output, per_sample_records)
        self._write_jsonl(self.step_trace_output, step_trace_records)
        print('--------------------')
        for i in range(batch_size):
            print(f'Response {i}:', responses[i])
            print('====================')
        print('--------------------')
        return responses
    
    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        raise NotImplementedError('Please use `lmeval` toolkit instead.')

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_ppl(inputs, mask_length=mask_length)
        else:
            if mask_length is not None:
                print('_______')
                return np.concatenate([
                    self._get_ppl(inputs=[text],
                                         mask_length=[mask_length[idx]])
                    for idx, text in enumerate(inputs)
                ])
            return np.concatenate([
                self._get_ppl(inputs=[text], mask_length=mask_length)
                for text in inputs
            ])

    def get_loglikelihood(self, inputs: List[str], conts:  List[str]) -> List[float]:
        print('inputs:', inputs, 'conts:', conts)
        mask_length = [self.get_token_len(c, add_special_tokens=False) for c in conts]
        return - self.get_ppl(inputs, mask_length)
    
    def get_logits(self, inputs, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == inputs.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(inputs.shape[0], 1)
            un_inputs = inputs.clone()
            un_inputs[prompt_index] = self.mask_id
            inputs = torch.cat([inputs, un_inputs])

        logits = self.model(inputs).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :inputs.shape[1]]

def _convert_chat_messages(inputs, merge_role=True, skip_empty_prompt=True):
    outputs = []
    for _input in inputs:
        messages = []
        if isinstance(_input, str):
            messages.append({'role': 'user', 'content': _input})
        else:
            for item in _input:
                if skip_empty_prompt and not item['prompt']:
                    continue
                role = {
                    'HUMAN': 'user',
                    'BOT': 'assistant',
                    'SYSTEM': 'system',
                }[item['role']]
                messages.append({'role': role, 'content': item['prompt']})

        if merge_role:
            merged_messages = []
            for item in messages:
                if merged_messages and merged_messages[-1]['role'] == item['role']:
                    merged_messages[-1]['content'] += '\n' + item['content']
                else:
                    merged_messages.append(item)
            messages = merged_messages

        outputs.append(messages)
    return outputs


def  _convert_base_messages(inputs):
    outputs = []
    for _input in inputs:
        if isinstance(_input, str):
            outputs.append(_input)
        else:
            messages = []
            for item in _input:
                messages.append(item['prompt'])
            outputs.append(''.join(messages))
    return outputs

class LLaDABaseModel(LLaDAModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.template_parser = LMTemplateParser()
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs. """
        messages = _convert_base_messages(inputs)
        context_prefix = self._context_prefix()
        prompt = [context_prefix + message for message in messages]
        print('steps:', self.gen_steps, 'length:', self.gen_length, 'blocksize:', self.gen_blocksize)
        print('temperature:', self.temperature, 'cfg:', self.cfg, 'remasking:', self.remasking)
        print('mask_id:', self.mask_id, 'padding_id:', self.padding_id)
        print('diff_confidence_eos_eot_inf:', self.diff_confidence_eos_eot_inf, 'diff_logits_eos_inf:', self.diff_logits_eos_inf)
        print('token_selection_confidence_threshold:', self.token_selection_confidence_threshold)
        print('min_transfer_tokens:', self.min_transfer_tokens, 'per_sample_output:', self.per_sample_output, 'step_trace_output:', self.step_trace_output, 'return_trace:', self.return_trace)
        print('trace_token_snapshots:', self.trace_token_snapshots, 'trace_decode_snapshots:', self.trace_decode_snapshots)
        print('final prompt:', prompt)
        prompt_texts = prompt
        self.tokenizer.padding_side = "left" 
        encoded_prompt = self.tokenizer.batch_encode_plus(prompt, padding = True, return_tensors='pt')
        prompt = encoded_prompt['input_ids']
        tokenized_prompts = prompt.detach().cpu().tolist()
        attention_mask = encoded_prompt.get('attention_mask')
        profile_trace = self.return_trace or bool(self.per_sample_output)
        self._cuda_stats_before()
        started = time.perf_counter()
        generated = LLaDA_generate(
            model = self.model,
            prompt = prompt.to(self.model.device),
            attention_mask = attention_mask.to(self.model.device) if attention_mask is not None else None,
            steps = self.gen_steps,
            gen_length = self.gen_length,
            block_length = self.gen_blocksize,
            temperature = self.temperature,
            cfg_scale = self.cfg,
            remasking = self.remasking,
            mask_id = self.mask_id,
            confidence_eos_eot_inf = self.diff_confidence_eos_eot_inf,
            logits_eos_inf = self.diff_logits_eos_inf,
            token_selection_confidence_threshold = self.token_selection_confidence_threshold,
            min_transfer_tokens = self.min_transfer_tokens,
            return_trace = profile_trace,
            trace_token_snapshots = self.trace_token_snapshots or self.trace_decode_snapshots,
            tokenizer = self.tokenizer,
        )
        elapsed = time.perf_counter() - started
        cuda_stats = self._cuda_stats_after()
        if profile_trace:
            x, trace = generated
        else:
            x, trace = generated, None
        responses = []
        batch_size = prompt.shape[0]
        
        for i in range(batch_size):
            responses.append(self.tokenizer.decode(x[i, -self.gen_length:], skip_special_tokens=True))
        print('--------------------')
        for i in range(batch_size):
            print(f'Response {i}:', responses[i])
            print('====================')
        print('--------------------')
        stopping_criteria = set(self.stop_words)
        for i in range(batch_size):
            response = responses[i]
            for stop_word in stopping_criteria:
                if stop_word in response:
                    response = response.split(stop_word)[0]
                    break
            responses[i] = response
        per_sample_records, step_trace_records = self._build_profile_records(
            prompt_texts=prompt_texts,
            tokenized_prompts=tokenized_prompts,
            responses=responses,
            prompt_tokens=prompt.shape[1],
            elapsed=elapsed,
            trace=trace,
            cuda_stats=cuda_stats,
        )
        self._write_jsonl(self.per_sample_output, per_sample_records)
        self._write_jsonl(self.step_trace_output, step_trace_records)
        return responses
    
    def get_token_len(self, prompt: str, add_special_tokens: bool=True) -> int:
        m = _convert_base_messages([prompt])[0]
        t = self.tokenizer(m, add_special_tokens=add_special_tokens)
        return len(t['input_ids'])
    
