"""Deterministic synthetic retrieval tasks. Scores are diagnostics, not a general benchmark."""
import hashlib
import json
import random
import re


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def exact_answer(text, expected):
    # Allow only wrapping whitespace, quotes and terminal punctuation. Additional
    # prose or a list containing the right answer does not count as exact retrieval.
    return text.strip().strip('"\'` .\n\r\t').casefold() == expected.casefold()


def build_case(tok, budget, position, seed, absent=False):
    if budget < 128 or not 0 <= position <= 1:
        raise ValueError('budget must be >=128 and position in [0,1]')
    rng = random.Random(seed)
    target = 'target-' + str(rng.randrange(100000, 999999))
    answer = 'CODE-' + ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(8))
    record = f'Record {target}: access code = {answer}.\n'
    system = ('Read the records and answer the question using only the records. '
              'Return only the access code, with no explanation. If the requested record is absent, return UNKNOWN.')
    question = f'\nQuestion: What is the access code for record {target}?'
    def render(count):
        rows = [f'Record item-{i:05d}: access code = CODE-{rng_seed_code(seed, i)}.\n' for i in range(count)]
        index = round(position * count)
        before = ''.join(rows[:index])
        context = before + ('' if absent else record) + ''.join(rows[index:])
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': 'Records:\n' + context + question}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(text, add_special_tokens=False)
        return messages, text, ids, before
    lo, hi = 0, budget
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(render(mid)[2]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    messages, text, ids, before = render(lo)
    if len(ids) > budget:
        raise ValueError('budget too small for the task instructions')
    offset = None if absent else len(tok.encode(text[:text.index(record)], add_special_tokens=False))
    return {'kind': 'absent' if absent else 'retrieval', 'budget_tokens': budget,
            'prompt_tokens': len(ids), 'position_requested': position,
            'needle_token_offset': offset,
            'needle_fraction_of_prompt': offset / len(ids) if offset is not None else None,
            'seed': seed, 'target_record': target, 'expected': 'UNKNOWN' if absent else answer,
            'messages': messages, 'prompt': text, 'input_ids': ids,
            'prompt_sha256': token_hash(ids)}


def rng_seed_code(seed, index):
    rng = random.Random(f'{seed}:{index}')
    return ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(8))


def build_followups(tok, case, assistant_text):
    """Paired prompts differ only in how the requested record is referred to."""
    notes = 'Additional irrelevant note: the delivery desk closes at noon. ' * 32
    questions = {
        'implicit': 'Now repeat the access code for the same requested record. Return only the code.',
        'explicit': f"Now repeat the access code for record {case['target_record']}. Return only the code.",
    }
    prompts = []
    for variant, question in questions.items():
        messages = [*case['messages'], {'role': 'assistant', 'content': assistant_text},
                    {'role': 'user', 'content': notes + question}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(text, add_special_tokens=False)
        prompts.append({'kind': 'followup', 'variant': variant, 'seed': case['seed'],
                        'target_record': case['target_record'], 'expected': case['expected'],
                        'initial_prompt_sha256': case['prompt_sha256'], 'assistant_context': assistant_text,
                        'messages': messages, 'prompt': text, 'input_ids': ids,
                        'prompt_tokens': len(ids), 'prompt_sha256': token_hash(ids)})
    return prompts
