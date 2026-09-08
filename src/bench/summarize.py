"""Summarize completed experiment JSON without inventing cross-backend equivalence."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(reports):
    lines = ['# CPU baseline and context evaluation', '',
             'These are measurements on one CPU host with a small synthetic task set. '
             'HF uses FP32 and GGUF uses Q4_K_M; both runtime and precision differ. '
             'This does not isolate a kernel or architecture speedup.', '']
    for report in reports:
        if not report.get('complete'):
            raise ValueError('refusing to summarize an incomplete run')
    baseline = [r for r in reports if r['mode'] == 'baseline']
    if baseline:
        lines += ['## Cold-prompt baseline', '',
                  'Model loading/tokenization are excluded. Each measured request resets its cache. '
                  'Warmup runs are excluded. TPOT covers intervals after the first sampled token. '
                  'RSS is sampled whole-process memory and may miss brief peaks.', '',
                  '| Backend | Prompt tokens | Trials | Output token range | Median TTFT (s) | TTFT stdev (s) | Median TPOT (s) | Peak sampled RSS (GiB) |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for report in baseline:
            grouped = defaultdict(list)
            for row in report['records']:
                if row['status'] == 'ok':
                    grouped[row['prompt_tokens']].append(row)
            for tokens, rows in grouped.items():
                times = [r['ttft_s'] for r in rows]
                tpots = [r['tpot_s'] for r in rows if r['tpot_s'] is not None]
                tpot = f'{statistics.median(tpots):.4f}' if tpots else 'N/A'
                sd = statistics.stdev(times) if len(times) > 1 else 0
                lines.append(f"| {report['settings']['backend']} | {tokens} | {len(rows)} | {min(r['generated_tokens'] for r in rows)}-{max(r['generated_tokens'] for r in rows)} | {statistics.median(times):.3f} | {sd:.3f} | {tpot} | {max(r['rss_peak_sampled_bytes'] for r in rows)/2**30:.3f} |")
        if len(baseline) == 2:
            a, b = baseline
            hashes = lambda r: {x['prompt_sha256'] for x in r['records'] if x['status'] == 'ok'}
            matched = hashes(a) == hashes(b) and all(a['settings'][k] == b['settings'][k] for k in ('context','threads'))
            matched = matched and a['parameters']['max_new'] == b['parameters']['max_new'] and a['spec'] == b['spec']
            lines += ['', f'Cross-backend prompt hashes and context/thread/output settings match: **{matched}**.']
            outputs = lambda r: {(x['prompt_sha256'], tuple(x['output_ids'])) for x in r['records'] if x['status'] == 'ok'}
            lines += [f'Observed greedy output token sequences match across backends: **{outputs(a) == outputs(b)}**.']
    quality = [r for r in reports if r['mode'] == 'quality']
    if quality:
        lines += ['', '## Retrieval quality', '',
                  'A strict exact-answer score allows surrounding whitespace/quotes/punctuation, but rejects extra prose. '
                  'The answer occurs once in a seeded distractor corpus. Absent-record cases test UNKNOWN responses. '
                  'A budget is an upper bound; whole records are retained and actual lengths are reported.', '',
                  '| Backend | Budget | Actual prompt range | Retrieval exact | Absent exact | Retrieval by requested position |',
                  '|---|---:|---:|---:|---:|---|']
        for report in quality:
            grouped = defaultdict(list)
            for row in report['records']:
                if row['status'] == 'ok':
                    grouped[row['budget_tokens']].append(row)
            for budget, rows in grouped.items():
                positive = [r for r in rows if r['kind'] == 'retrieval']
                negative = [r for r in rows if r['kind'] == 'absent']
                positions = ', '.join(f"{p:.0%}: {sum(r['exact_match'] for r in positive if r['position_requested']==p)}/{sum(r['position_requested']==p for r in positive)}" for p in sorted({r['position_requested'] for r in positive}))
                lines.append(f"| {report['settings']['backend']} | {budget} | {min(r['prompt_tokens'] for r in rows)}-{max(r['prompt_tokens'] for r in rows)} | {sum(r['exact_match'] for r in positive)}/{len(positive)} | {sum(r['exact_match'] for r in negative)}/{len(negative)} | {positions} |")
    failures = [(report['settings']['backend'], row) for report in quality for row in report['records']
                if row.get('status') == 'ok' and not row['exact_match']]
    if failures:
        lines += ['', 'Observed exact-answer failures:', '']
        for backend, row in failures:
            lines.append(f"- {backend}, {row['prompt_tokens']} tokens, {row['kind']}, seed {row['seed']}: expected `{row['expected']}`, received `{row['text'].strip()}`.")
    sessions = [r for r in reports if r['mode'] == 'sessions']
    if sessions:
        lines += ['', '## Growing conversations', '',
                  'Each turn compares cached and freshly reset execution of identical prompt IDs. '
                  'The next turn extends the fresh control conversation. These are single observations, not repeated speedup estimates.', '',
                  '| Backend | Turn | Prompt tokens | Reused tokens | Cached TTFT (s) | Fresh TTFT (s) | Same output IDs | Exact answer |',
                  '|---|---:|---:|---:|---:|---:|---|---|']
        for report in sessions:
            for r in report['records']:
                if r['status'] == 'ok':
                    lines.append(f"| {report['settings']['backend']} | {r['turn']} | {r['prompt_tokens']} | {r['warm']['reused_tokens']} | {r['warm']['ttft_s']:.3f} | {r['cold']['ttft_s']:.3f} | {r['outputs_equal']} | {r['exact_match']} |")
    lines += ['', '## Reproducibility and limits', '']
    if reports:
        env = reports[0]['environment']
        lines += [f"Host: {env['cpu']}; {env['physical_cpus']} physical / {env['logical_cpus']} logical CPUs; {env['ram_bytes']/2**30:.2f} GiB RAM.", '',
                  'Model revisions, artifact hashes, prompts/token IDs, seeds, generation outputs, native CPU features, '
                  'dependency versions, settings and per-case timing are preserved in the raw JSON. '
                  'The harness is identified by source hashes even when run before its commit.', '',
                  'Supported model context is 32,768 tokens. The tested lengths establish neither 32K nor 50K quality. '
                  'One seed per position is a diagnostic sample, not a statistically reliable quality estimate. '
                  'Whole-process RSS includes shared Python/PyTorch imports in both runners and is not isolated KV memory.']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('reports', nargs='+')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    reports = [json.loads(Path(p).read_text(encoding='utf-8')) for p in args.reports]
    Path(args.output).write_text(summarize(reports), encoding='utf-8')
