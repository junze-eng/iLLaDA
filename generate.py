import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Optional

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    # float64 here reliably OOMs on 24GB cards at this vocab size (155136) once the
    # base model + activations already use ~22GB -- float32 halves that overhead.
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
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


def resolve_steps_per_block_schedule(
        *,
        speed_schedule_name: Optional[str] = None,
        steps_per_block_schedule: Optional[List[int]] = None,
        steps: int,
        gen_length: int,
        block_length: int):
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    if speed_schedule_name is not None and steps_per_block_schedule is not None:
        raise ValueError("Pass either speed_schedule_name or steps_per_block_schedule, not both.")

    schedule_overrides_gen_steps = speed_schedule_name is not None or steps_per_block_schedule is not None
    if speed_schedule_name is not None:
        gsm8k_pilot_schedules = {
            'all_1tps': [32, 32, 32, 32, 32, 32, 32, 32],
            'all_2tps': [16, 16, 16, 16, 16, 16, 16, 16],
            'all_4tps': [8, 8, 8, 8, 8, 8, 8, 8],
            'slow_to_fast_1_2_4': [32, 32, 16, 16, 16, 8, 8, 8],
            'fast_to_slow_4_2_1': [8, 8, 8, 16, 16, 16, 32, 32],
            'slow_fast_slow_1_4_1': [32, 32, 8, 8, 8, 8, 32, 32],
        }
        if gen_length != 256 or block_length != 32 or num_blocks != 8:
            raise ValueError(
                "speed_schedule_name currently supports only the GSM8K pilot "
                f"(gen_length=256, block_length=32, num_blocks=8); got "
                f"gen_length={gen_length}, block_length={block_length}, num_blocks={num_blocks}."
            )
        if speed_schedule_name not in gsm8k_pilot_schedules:
            raise ValueError(
                f"Unknown speed_schedule_name `{speed_schedule_name}`. "
                f"Available: {', '.join(sorted(gsm8k_pilot_schedules))}"
            )
        steps_per_block_schedule = gsm8k_pilot_schedules[speed_schedule_name]
    elif steps_per_block_schedule is None:
        assert steps % num_blocks == 0
        uniform_steps_per_block = steps // num_blocks
        steps_per_block_schedule = [uniform_steps_per_block] * num_blocks

    block_steps_schedule = [int(block_steps) for block_steps in steps_per_block_schedule]
    if len(block_steps_schedule) != num_blocks:
        raise ValueError(
            f"steps_per_block_schedule must have {num_blocks} entries, got {len(block_steps_schedule)}."
        )
    if any(block_steps <= 0 for block_steps in block_steps_schedule):
        raise ValueError(f"steps_per_block_schedule values must be positive integers: {block_steps_schedule}")

    planned_steps = sum(block_steps_schedule)
    return block_steps_schedule, planned_steps, schedule_overrides_gen_steps


@ torch.no_grad()
def _dynamic_stop_token_ids(tokenizer=None, stop_token_ids=None):
    ids = []
    if stop_token_ids is not None:
        ids.extend(stop_token_ids)
    if tokenizer is not None:
        candidates = [
            getattr(tokenizer, 'eos_token_id', None),
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            tokenizer.convert_tokens_to_ids("</think>"),
        ]
        unk_id = getattr(tokenizer, 'unk_token_id', None)
        vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else None
        for token_id in candidates:
            if token_id is None:
                continue
            if token_id == unk_id:
                continue
            if vocab_size is not None and not (0 <= int(token_id) < vocab_size):
                continue
            ids.append(int(token_id))
    return sorted(set(int(token_id) for token_id in ids if token_id is not None))


@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False,
             confidence_eos_eot_inf=False, token_selection_confidence_threshold=None,
             min_transfer_tokens=1, return_trace=False, trace_token_snapshots=False,
             tokenizer=None, stop_token_ids=None, speed_schedule_name: Optional[str] = None,
             steps_per_block_schedule: Optional[List[int]] = None,
             token_selection_confidence_threshold_schedule: Optional[List[Optional[float]]] = None,
             trace_step0_full_confidence: bool = False,
             decode_order: str = 'confidence'):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        token_selection_confidence_threshold: If set, only predicted tokens with confidence greater than or equal to
            this threshold are transferred. This is useful for ARness ablations because a higher threshold makes each
            diffusion step commit fewer tokens.
        min_transfer_tokens: Minimum number of tokens to transfer for each sample when the scheduled transfer count is
            positive. This prevents high confidence thresholds from stalling with residual mask tokens.
        return_trace: If True, return (tokens, trace) with per-step transfer statistics.
        trace_token_snapshots: If True, include generated-region token IDs after every diffusion step.
        speed_schedule_name: Optional named per-block speed schedule.
        steps_per_block_schedule: Optional explicit denoising-step count for each block.
        token_selection_confidence_threshold_schedule: Optional per-block confidence threshold schedule.
        trace_step0_full_confidence: If True (and return_trace=True), record the model's confidence for
            every generation-region position (not just selected ones) at the very first diffusion step,
            to study how initial confidence decays with distance from the prompt.
        decode_order: Which positions get committed each step. 'confidence' (default) picks the
            highest-confidence masked positions, as usual. 'left_to_right' forces the leftmost masked
            positions to be committed regardless of confidence (still using the model's own predicted
            token for that position). 'random' picks a random subset of masked positions. The latter two
            exist to measure whether iLLaDA's free (confidence-based) decoding order carries real value
            over a forced order, by comparing accuracy across all three at matched steps/gen_length.
    '''
    stop_token_ids = _dynamic_stop_token_ids(tokenizer=tokenizer, stop_token_ids=stop_token_ids)

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    block_steps_schedule, planned_steps, schedule_overrides_gen_steps = resolve_steps_per_block_schedule(
        speed_schedule_name=speed_schedule_name,
        steps_per_block_schedule=steps_per_block_schedule,
        steps=int(steps),
        gen_length=int(gen_length),
        block_length=int(block_length),
    )
    if schedule_overrides_gen_steps:
        print(
            'speed_schedule:',
            speed_schedule_name,
            'schedule:',
            block_steps_schedule,
            'planned_steps:',
            planned_steps,
        )
    num_blocks = gen_length // block_length
    uniform_steps_per_block = block_steps_schedule[0] if len(set(block_steps_schedule)) == 1 else None
    threshold_schedule = None
    if token_selection_confidence_threshold_schedule is not None:
        if len(token_selection_confidence_threshold_schedule) != num_blocks:
            raise ValueError(
                "token_selection_confidence_threshold_schedule must have "
                f"{num_blocks} entries, got {len(token_selection_confidence_threshold_schedule)}."
            )
        threshold_schedule = [
            None if threshold is None else float(threshold)
            for threshold in token_selection_confidence_threshold_schedule
        ]
    schedule_overrides_scalar_threshold = (
        threshold_schedule is not None and token_selection_confidence_threshold is not None
    )
    if schedule_overrides_scalar_threshold:
        print(
            "token_selection_confidence_threshold_schedule overrides "
            f"token_selection_confidence_threshold={token_selection_confidence_threshold}"
        )

    trace = {
        'prompt_tokens': int(prompt.shape[1]),
        'gen_length': int(gen_length),
        'block_length': int(block_length),
        'steps_per_block': int(uniform_steps_per_block) if uniform_steps_per_block is not None else None,
        'steps_per_block_schedule': list(block_steps_schedule),
        'speed_schedule_name': speed_schedule_name,
        'planned_steps': int(planned_steps),
        'forward_passes': int(planned_steps),
        'num_blocks': int(num_blocks),
        'planned_parallelism': float(gen_length / planned_steps) if planned_steps else None,
        'schedule_overrides_gen_steps': bool(schedule_overrides_gen_steps),
        'mask_id': int(mask_id),
        'stop_token_ids': stop_token_ids,
        'token_selection_confidence_threshold': token_selection_confidence_threshold,
        'token_selection_confidence_threshold_schedule': threshold_schedule,
        'schedule_overrides_scalar_threshold': bool(schedule_overrides_scalar_threshold),
        'block_thresholds': threshold_schedule if threshold_schedule is not None else [token_selection_confidence_threshold] * num_blocks,
        'min_transfer_tokens': int(min_transfer_tokens),
        'scheduled_transfer_count': 0,
        'threshold_passed_count': 0,
        'fallback_forced_count': 0,
        'actual_transfer_count': 0,
        'step_stats': [],
        'token_snapshots': [] if trace_token_snapshots else None,
        'step0_confidence_by_position': [] if trace_step0_full_confidence else None,
    } if return_trace else None

    global_step_offset = 0
    for num_block in range(num_blocks):
        block_steps = block_steps_schedule[num_block]
        block_threshold = threshold_schedule[num_block] if threshold_schedule is not None else token_selection_confidence_threshold
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, block_steps)
        for i in range(block_steps):
            step_idx = global_step_offset + i
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf and stop_token_ids:
                logits[:, :, stop_token_ids] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            if confidence_eos_eot_inf and stop_token_ids:
                stop_predictions = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for token_id in stop_token_ids:
                    stop_predictions |= x0 == token_id
                confidence = torch.where(stop_predictions, torch.full_like(confidence, -torch.inf), confidence)

            if trace_step0_full_confidence and trace is not None and step_idx == 0:
                gen_region_confidence = confidence[:, prompt.shape[1]:]
                trace['step0_confidence_by_position'] = gen_region_confidence.detach().cpu().tolist()

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            step_scheduled_count = 0
            step_passed_count = 0
            step_forced_count = 0
            step_actual_count = 0
            step_confidence_sum = 0.0
            step_records = []
            mask_count_before = int((x[:, prompt.shape[1]:] == mask_id).sum().item()) if trace is not None else None
            for j in range(confidence.shape[0]):
                scheduled_count = int(num_transfer_tokens[j, i].item())
                if scheduled_count <= 0:
                    continue
                step_scheduled_count += scheduled_count
                if decode_order == 'confidence':
                    select_confidence, select_index = torch.topk(confidence[j], k=scheduled_count)
                elif decode_order == 'left_to_right':
                    candidate_positions = torch.isfinite(confidence[j]).nonzero(as_tuple=True)[0]
                    select_index = candidate_positions[:scheduled_count]
                    select_confidence = confidence[j][select_index]
                elif decode_order == 'random':
                    candidate_positions = torch.isfinite(confidence[j]).nonzero(as_tuple=True)[0]
                    perm = candidate_positions[torch.randperm(candidate_positions.numel(), device=candidate_positions.device)]
                    select_index = perm[:scheduled_count]
                    select_confidence = confidence[j][select_index]
                else:
                    raise ValueError(f"Unknown decode_order `{decode_order}`. Expected 'confidence', 'left_to_right', or 'random'.")
                original_select_confidence = select_confidence
                original_select_index = select_index
                passed_count = scheduled_count
                forced_count = 0
                transfer_reason = 'scheduled'
                if block_threshold is not None:
                    keep = select_confidence >= block_threshold
                    passed_count = int(keep.sum().item())
                    if keep.sum().item() == 0 and min_transfer_tokens > 0:
                        keep[:min(min_transfer_tokens, scheduled_count)] = True
                        forced_count = int(keep.sum().item())
                        transfer_reason = 'threshold_fallback_forced'
                    else:
                        transfer_reason = 'threshold_pass'
                    select_confidence = select_confidence[keep]
                    select_index = select_index[keep]
                transfer_index[j, select_index] = True
                actual_count = int(select_index.numel())
                step_passed_count += passed_count
                step_forced_count += forced_count
                step_actual_count += actual_count
                if select_confidence.numel() > 0:
                    step_confidence_sum += float(select_confidence.float().sum().item())
                if trace is not None:
                    prompt_len = int(prompt.shape[1])
                    selected_positions = [
                        int(pos) - prompt_len for pos in select_index.detach().cpu().tolist()
                        if int(pos) >= prompt_len
                    ]
                    selected_token_ids = [
                        int(x0[j, int(pos)].detach().cpu().item())
                        for pos in select_index.detach().cpu().tolist()
                        if int(pos) >= prompt_len
                    ]
                    selected_confidences = [
                        float(value)
                        for value in select_confidence.detach().float().cpu().tolist()
                    ]
                    step_records.append({
                        'batch_item_idx': int(j),
                        'block_idx': int(num_block),
                        'block_steps': int(block_steps),
                        'block_planned_parallelism': float(block_length / block_steps) if block_steps else None,
                        'block_threshold': block_threshold,
                        'step_idx': int(step_idx),
                        'global_step_idx': int(step_idx),
                        'step_idx_in_block': int(i),
                        'local_step_idx': int(i),
                        'mask_count_before': None,
                        'selected_positions': selected_positions,
                        'selected_token_ids': selected_token_ids,
                        'selected_confidences': selected_confidences,
                        'transfer_reason': transfer_reason,
                        'scheduled_transfer_count': int(scheduled_count),
                        'threshold_passed_count': int(passed_count),
                        'fallback_forced_count': int(forced_count),
                        'actual_transfer_count': int(actual_count),
                        'candidate_positions': [
                            int(pos) - prompt_len for pos in original_select_index.detach().cpu().tolist()
                            if int(pos) >= prompt_len
                        ],
                        'candidate_confidences': [
                            float(value)
                            for value in original_select_confidence.detach().float().cpu().tolist()
                        ],
                    })
            x[transfer_index] = x0[transfer_index]
            if trace is not None:
                generated_region = x[:, prompt.shape[1]:]
                mask_count_after = int((generated_region == mask_id).sum().item())
                trace['scheduled_transfer_count'] += int(step_scheduled_count)
                trace['threshold_passed_count'] += int(step_passed_count)
                trace['fallback_forced_count'] += int(step_forced_count)
                trace['actual_transfer_count'] += int(step_actual_count)
                if not step_records:
                    step_records.append({
                        'batch_item_idx': 0,
                        'block_idx': int(num_block),
                        'block_steps': int(block_steps),
                        'block_planned_parallelism': float(block_length / block_steps) if block_steps else None,
                        'block_threshold': block_threshold,
                        'step_idx': int(step_idx),
                        'global_step_idx': int(step_idx),
                        'step_idx_in_block': int(i),
                        'local_step_idx': int(i),
                        'mask_count_before': mask_count_before,
                        'selected_positions': [],
                        'selected_token_ids': [],
                        'selected_confidences': [],
                        'transfer_reason': 'no_scheduled_transfer',
                        'scheduled_transfer_count': 0,
                        'threshold_passed_count': 0,
                        'fallback_forced_count': 0,
                        'actual_transfer_count': 0,
                    })
                for step_record in step_records:
                    item_idx = int(step_record.get('batch_item_idx', 0))
                    item_region = generated_region[item_idx:item_idx + 1]
                    block_start = num_block * block_length
                    block_end = (num_block + 1) * block_length
                    item_block = item_region[:, block_start:block_end]
                    block_masks_after = int((item_block == mask_id).sum().item())
                    block_visible_tokens = int(item_block.numel() - block_masks_after)
                    item_mask_after = int((item_region == mask_id).sum().item())
                    item_total = int(item_region.numel())
                    step_record['mask_count_before'] = mask_count_before
                    step_record['mask_count_after'] = item_mask_after
                    step_record['remaining_masks'] = item_mask_after
                    step_record['block_visible_tokens'] = block_visible_tokens
                    step_record['block_completion_rate'] = block_visible_tokens / max(int(item_block.numel()), 1)
                    step_record['cumulative_transferred_tokens'] = int(trace['actual_transfer_count'])
                    step_record['current_completion_rate'] = (item_total - item_mask_after) / max(item_total, 1)
                    step_record['mean_confidence'] = (
                        sum(step_record['selected_confidences']) / len(step_record['selected_confidences'])
                        if step_record.get('selected_confidences') else None
                    )
                    trace['step_stats'].append(step_record)
                if trace_token_snapshots:
                    trace['token_snapshots'].append({
                        'block_id': int(num_block),
                        'step_id': int(i),
                        'global_step_id': int(step_idx),
                        'generated_token_ids': x[:, prompt.shape[1]:].detach().cpu().tolist(),
                    })
        global_step_offset += block_steps

    if trace is not None:
        generated_region = x[:, prompt.shape[1]:]
        final_mask_count = int((generated_region == mask_id).sum().item())
        total_gen_tokens = int(generated_region.numel())
        visible_tokens_by_block = []
        completion_rate_by_block = []
        for batch_idx in range(generated_region.shape[0]):
            sample_visible = []
            sample_completion = []
            for block_idx in range(num_blocks):
                block_start = block_idx * block_length
                block_end = (block_idx + 1) * block_length
                block_region = generated_region[batch_idx, block_start:block_end]
                visible = int((block_region != mask_id).sum().item())
                sample_visible.append(visible)
                sample_completion.append(visible / max(int(block_region.numel()), 1))
            visible_tokens_by_block.append(sample_visible)
            completion_rate_by_block.append(sample_completion)
        trace['visible_tokens_by_block'] = visible_tokens_by_block
        trace['completion_rate_by_block'] = completion_rate_by_block
        trace['final_mask_count'] = final_mask_count
        trace['completion_rate'] = (total_gen_tokens - final_mask_count) / max(total_gen_tokens, 1)
        trace['actual_parallelism'] = trace['actual_transfer_count'] / max(trace['forward_passes'], 1)
        trace['actual_arness'] = trace['forward_passes'] / max(trace['actual_transfer_count'], 1)
        trace['threshold_pass_rate'] = trace['threshold_passed_count'] / max(trace['scheduled_transfer_count'], 1)
        trace['fallback_rate'] = trace['fallback_forced_count'] / max(trace['actual_transfer_count'], 1)
    return (x, trace) if return_trace else x


def main():
    device = 'cuda'
    model_path = 'GSAI-ML/iLLaDA-8B-Instruct'
    mask_id = 5

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != mask_id

    prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
             "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
             "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32,
                   temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=mask_id,
                   tokenizer=tokenizer)
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()
