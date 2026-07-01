import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


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


def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False,
             confidence_eos_eot_inf=False, token_selection_confidence_threshold=None,
             min_transfer_tokens=1, return_trace=False, trace_token_snapshots=False,
             tokenizer=None, stop_token_ids=None):
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
    '''
    planned_steps = int(steps)
    stop_token_ids = _dynamic_stop_token_ids(tokenizer=tokenizer, stop_token_ids=stop_token_ids)

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    trace = {
        'prompt_tokens': int(prompt.shape[1]),
        'gen_length': int(gen_length),
        'block_length': int(block_length),
        'steps_per_block': int(steps),
        'planned_steps': int(planned_steps),
        'forward_passes': int(planned_steps),
        'num_blocks': int(num_blocks),
        'planned_parallelism': float(gen_length / planned_steps) if planned_steps else None,
        'mask_id': int(mask_id),
        'stop_token_ids': stop_token_ids,
        'token_selection_confidence_threshold': token_selection_confidence_threshold,
        'min_transfer_tokens': int(min_transfer_tokens),
        'scheduled_transfer_count': 0,
        'threshold_passed_count': 0,
        'fallback_forced_count': 0,
        'actual_transfer_count': 0,
        'step_stats': [],
        'token_snapshots': [] if trace_token_snapshots else None,
    } if return_trace else None

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
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

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            step_scheduled_count = 0
            step_passed_count = 0
            step_forced_count = 0
            step_actual_count = 0
            step_confidence_sum = 0.0
            for j in range(confidence.shape[0]):
                scheduled_count = int(num_transfer_tokens[j, i].item())
                if scheduled_count <= 0:
                    continue
                step_scheduled_count += scheduled_count
                select_confidence, select_index = torch.topk(confidence[j], k=scheduled_count)
                passed_count = scheduled_count
                forced_count = 0
                if token_selection_confidence_threshold is not None:
                    keep = select_confidence >= token_selection_confidence_threshold
                    passed_count = int(keep.sum().item())
                    if keep.sum().item() == 0 and min_transfer_tokens > 0:
                        keep[:min(min_transfer_tokens, scheduled_count)] = True
                        forced_count = int(keep.sum().item())
                    select_confidence = select_confidence[keep]
                    select_index = select_index[keep]
                transfer_index[j, select_index] = True
                actual_count = int(select_index.numel())
                step_passed_count += passed_count
                step_forced_count += forced_count
                step_actual_count += actual_count
                if select_confidence.numel() > 0:
                    step_confidence_sum += float(select_confidence.float().sum().item())
            x[transfer_index] = x0[transfer_index]
            if trace is not None:
                generated_region = x[:, prompt.shape[1]:]
                trace['scheduled_transfer_count'] += int(step_scheduled_count)
                trace['threshold_passed_count'] += int(step_passed_count)
                trace['fallback_forced_count'] += int(step_forced_count)
                trace['actual_transfer_count'] += int(step_actual_count)
                trace['step_stats'].append({
                    'block_id': int(num_block),
                    'step_id': int(i),
                    'scheduled': int(step_scheduled_count),
                    'passed': int(step_passed_count),
                    'forced': int(step_forced_count),
                    'actual': int(step_actual_count),
                    'transferred_tokens': int(step_actual_count),
                    'mean_confidence': step_confidence_sum / step_actual_count if step_actual_count else None,
                    'remaining_masks': int((generated_region == mask_id).sum().item()),
                })
                if trace_token_snapshots:
                    trace['token_snapshots'].append({
                        'block_id': int(num_block),
                        'step_id': int(i),
                        'generated_token_ids': x[:, prompt.shape[1]:].detach().cpu().tolist(),
                    })

    if trace is not None:
        generated_region = x[:, prompt.shape[1]:]
        final_mask_count = int((generated_region == mask_id).sum().item())
        total_gen_tokens = int(generated_region.numel())
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
