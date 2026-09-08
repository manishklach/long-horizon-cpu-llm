"""Predeclared multi-seed recall controls; all generations use cold KV state."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from .evaluate import save
from .runners import Runner, environment
from .tasks import build_case, build_followups, exact_answer


def run(protocol, spec, models, output):
    if len(set(protocol['seeds'])) != len(protocol['seeds']) or not protocol['seeds']:
        raise ValueError('seeds must be nonempty and unique')
    if protocol['expected_cases'] != 4 * len(protocol['seeds']):
        raise ValueError('expected_cases must equal four per seed')
    runner = Runner(protocol['backend'], spec, models, protocol['context'],
                    protocol['threads'], protocol['batch'])
    report = {'schema_version': 1, 'mode': 'recall_controls', 'protocol': protocol,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'spec': spec,
              'settings': runner.settings, 'manifest': runner.manifest,
              'environment': environment(), 'records': [], 'complete': False}
    save(output, report)
    def execute(case, **extra):
        runner.check_prompt(case)
        result = runner.generate(case['input_ids'], protocol['max_new'], cold=True)
        row = {**case, **result, **extra, 'status': 'ok',
               'exact_match': exact_answer(result['text'], case['expected'])}
        report['records'].append(row)
        save(output, report)
        print(f"seed {case['seed']} {case['kind']} {case.get('variant', '')}: "
              f"{row['exact_match']} ({result['ttft_s']:.2f}s) answer={result['text']!r}", flush=True)
        return row
    for index, seed in enumerate(protocol['seeds']):
        initial = build_case(runner.tok, protocol['initial_budget'], protocol['position'], seed)
        initial_result = execute(initial)
        pair = build_followups(runner.tok, initial, initial_result['text'])
        if index % 2:
            pair.reverse()
        for case in pair:
            execute(case, initial_exact_match=initial_result['exact_match'])
        execute(build_case(runner.tok, protocol['absent_budget'], .5, seed, absent=True))
    assert len(report['records']) == protocol['expected_cases']
    report['complete'] = True
    save(output, report)
    return report


def summarize(report):
    if not report.get('complete'):
        raise ValueError('incomplete report')
    lines = ['# Multi-seed recall controls', '',
             'Every generation starts with a fresh KV cache. Paired follow-ups share the exact '
             'initial transcript and actual assistant response; only the record reference changes.', '',
             '| Seed | Initial retrieval | Implicit follow-up | Explicit ID follow-up | Absent-record control |',
             '|---:|---|---|---|---|']
    def label(row):
        return ('PASS' if row['exact_match'] else 'FAIL') + ': `' + row['text'].strip() + '`'
    for seed in report['protocol']['seeds']:
        rows = [r for r in report['records'] if r['seed'] == seed]
        initial = next(r for r in rows if r['kind'] == 'retrieval')
        implicit = next(r for r in rows if r.get('variant') == 'implicit')
        explicit = next(r for r in rows if r.get('variant') == 'explicit')
        absent = next(r for r in rows if r['kind'] == 'absent')
        lines.append(f'| {seed} | {label(initial)} | {label(implicit)} | {label(explicit)} | {label(absent)} |')
    lines += ['', '## Counts', '']
    for variant in ('implicit', 'explicit'):
        eligible = [r for r in report['records'] if r.get('variant') == variant and r['initial_exact_match']]
        lines.append(f"- {variant} follow-up, conditional on correct initial retrieval: {sum(r['exact_match'] for r in eligible)}/{len(eligible)}.")
    absent = [r for r in report['records'] if r['kind'] == 'absent']
    lines += [f"- Absent-record controls: {sum(r['exact_match'] for r in absent)}/{len(absent)}.", '',
              'This is a small synthetic diagnostic on one model, quantization and host. Seed 17 is a '
              'replication; 29 and 41 are new. These results do not establish general reliability or '
              'a causal explanation beyond the tested prompt contrast. Raw prompts, token IDs, '
              'outputs, seeds, hashes and timing are preserved in the accompanying JSON.']
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', default='experiments/recall-controls-v1.json')
    ap.add_argument('--spec', default='experiments/qwen2.5-0.5b.json')
    ap.add_argument('--models', default='models')
    ap.add_argument('--output', default='data/experiments/recall-controls.json')
    ap.add_argument('--summarize-only', action='store_true')
    args = ap.parse_args()
    if args.summarize_only:
        report = json.loads(Path(args.output).read_text(encoding='utf-8'))
    else:
        protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8-sig'))
        spec = json.loads(Path(args.spec).read_text(encoding='utf-8-sig'))
        report = run(protocol, spec, args.models, args.output)
    Path(args.output).with_suffix('.md').write_text(summarize(report), encoding='utf-8')


if __name__ == '__main__':
    main()
