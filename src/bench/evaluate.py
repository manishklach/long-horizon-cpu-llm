"""CLI for pinned baseline, retrieval-position and growing-session experiments."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from .runners import Runner, prepare, environment
from .tasks import build_case, exact_answer, token_hash


def save(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(path)


def run(args):
    spec = json.loads(Path(args.spec).read_text(encoding='utf-8-sig'))
    if args.mode == 'prepare':
        prepare(spec, args.models)
        print('Pinned artifacts downloaded and checksummed.', flush=True)
        return
    runner = Runner(args.backend, spec, args.models, args.context, args.threads, args.batch)
    report = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'mode': args.mode, 'spec': spec, 'settings': runner.settings,
              'manifest': runner.manifest, 'environment': environment(),
              'parameters': vars(args), 'records': [], 'complete': False,
              'measurement_note': 'Token-level TTFT excludes tokenization/model loading; whole-process RSS sampled every 5ms.'}
    save(args.output, report)
    lengths = [int(x) for x in args.lengths.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    if not lengths or min(lengths) < 128 or args.repeats < 1 or args.warmups < 0:
        raise ValueError('lengths >=128, repeats >=1 and warmups >=0 required')
    for budget in lengths:
        if budget + args.max_new > args.context:
            report['records'].append({'budget_tokens': budget, 'status': 'skipped',
                                      'reason': 'requested prompt budget plus output exceeds configured context'})
            save(args.output, report)
            continue
        if args.mode == 'baseline':
            case = build_case(runner.tok, budget, 0.5, seeds[0])
            runner.check_prompt(case)
            for trial in range(args.warmups + args.repeats):
                result = runner.generate(case['input_ids'], args.max_new)
                if trial >= args.warmups:
                    record = {'trial': trial - args.warmups, 'status': 'ok', **case, **result}
                    report['records'].append(record)
                    save(args.output, report)
                    print(f"baseline {args.backend} {len(case['input_ids'])} tokens trial {trial-args.warmups}: TTFT {result['ttft_s']:.3f}s", flush=True)
        elif args.mode == 'quality':
            for seed in seeds:
                for position, absent in [(0.1, False), (0.5, False), (0.9, False), (0.5, True)]:
                    case = build_case(runner.tok, budget, position, seed, absent)
                    runner.check_prompt(case)
                    result = runner.generate(case['input_ids'], args.max_new)
                    record = {'status': 'ok', **case, **result, 'exact_match': exact_answer(result['text'], case['expected'])}
                    report['records'].append(record)
                    save(args.output, report)
                    print(f"quality {args.backend} {case['prompt_tokens']} tokens {case['kind']} {position}: {record['exact_match']} ({result['ttft_s']:.2f}s)", flush=True)
        elif args.mode == 'sessions':
            # Growing conversation uses full authoritative messages, templated
            # afresh each turn. Compare warm to fresh execution of EXACT same IDs.
            case = build_case(runner.tok, budget, 0.1, seeds[0])
            messages = case['messages']
            runner.reset()
            for turn in range(3):
                prompt = runner.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                ids = runner.tok.encode(prompt, add_special_tokens=False)
                runner.check_prompt({'prompt': prompt, 'input_ids': ids})
                if len(ids) + args.max_new > args.context:
                    report['records'].append({'status': 'skipped', 'turn': turn+1,
                                               'reason': 'growing conversation exceeds context limit'})
                    break
                warm = runner.generate(ids, args.max_new, cold=turn == 0)
                # Cold control changes cache state, so replay the warm result's
                # exact history on the next iteration via the identical cold output.
                cold = runner.generate(ids, args.max_new, cold=True)
                equal = warm['output_ids'] == cold['output_ids']
                report['records'].append({'status': 'ok', 'turn': turn+1, 'budget_tokens': budget,
                                          'prompt_tokens': len(ids), 'prompt_sha256': token_hash(ids),
                                          'prompt': prompt, 'input_ids': ids, 'expected': case['expected'],
                                          'warm': warm, 'cold': cold, 'outputs_equal': equal,
                                          'exact_match': exact_answer(warm['text'], case['expected'])})
                save(args.output, report)
                print(f"session {args.backend} turn {turn+1}: {len(ids)} tokens reused {warm['reused_tokens']}, equal={equal}", flush=True)
                messages = [*messages, {'role': 'assistant', 'content': cold['text']},
                            {'role': 'user', 'content': ('Additional irrelevant note: the delivery desk closes at noon. ' * 32)
                             + 'Now repeat the access code for the same requested record. Return only the code.'}]
    report['complete'] = True
    save(args.output, report)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['prepare', 'baseline', 'quality', 'sessions'])
    ap.add_argument('--spec', default='experiments/qwen2.5-0.5b.json')
    ap.add_argument('--models', default='models')
    ap.add_argument('--backend', choices=['hf', 'gguf'], default='gguf')
    ap.add_argument('--context', type=int, default=16384)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lengths', default='512,4096,8192')
    ap.add_argument('--max-new', type=int, default=24)
    ap.add_argument('--seeds', default='17')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--warmups', type=int, default=1)
    ap.add_argument('--output', default='data/evaluation.json')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
