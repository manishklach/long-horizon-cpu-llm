"""Pinned, token-level runners for experiments, separate from the serving API."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import psutil
from transformers import AutoTokenizer
from .benchmark import measured


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(spec, root):
    from huggingface_hub import snapshot_download, hf_hub_download
    root = Path(root).resolve()
    # slug selects per-size artifact dirs; default preserves the original layout.
    slug = spec.get('slug', 'qwen2.5-0.5b')
    hf = snapshot_download(spec['hf_repo'], revision=spec['hf_revision'], local_dir=root / f'{slug}-hf',
                           allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])
    gguf = hf_hub_download(spec['gguf_repo'], spec['gguf_file'], revision=spec['gguf_revision'],
                           local_dir=root / f'{slug}-gguf')
    if sha256(gguf) != spec['gguf_sha256']:
        raise ValueError('GGUF checksum mismatch')
    files = {p.resolve().relative_to(root).as_posix(): sha256(p) for p in sorted(Path(hf).glob('*')) if p.is_file()}
    files[Path(gguf).resolve().relative_to(root).as_posix()] = spec['gguf_sha256']
    manifest = {'spec': spec, 'files': files}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def verify(spec, root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['spec'] != spec:
        raise ValueError('manifest does not match experiment specification; run prepare')
    for name, expected in manifest['files'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'checksum mismatch: {name}')
    return manifest


def environment():
    packages = {}
    for package in ('torch', 'transformers', 'llama-cpp-python', 'numpy', 'psutil', 'huggingface-hub'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {'platform': platform.platform(), 'python': platform.python_version(),
            'cpu': platform.processor(), 'logical_cpus': psutil.cpu_count(),
            'physical_cpus': psutil.cpu_count(logical=False), 'ram_bytes': psutil.virtual_memory().total,
            'packages': packages, 'git_commit': git('rev-parse', 'HEAD'),
            'working_tree_dirty': bool(git('status', '--porcelain')),
            'harness_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}


class Runner:
    def __init__(self, backend, spec, root, context=16384, threads=4, batch=512):
        if min(context, threads, batch) <= 0 or context > spec['context_limit']:
            raise ValueError('invalid context, thread or batch settings')
        self.manifest = verify(spec, root)
        self.backend, self.context = backend, context
        slug = spec.get('slug', 'qwen2.5-0.5b')
        self.tok = AutoTokenizer.from_pretrained(Path(root) / f'{slug}-hf', local_files_only=True)
        self.settings = {'backend': backend, 'context': context, 'threads': threads, 'batch': batch,
                         'temperature': 0, 'repeat_penalty': 1, 'gpu_layers': 0}
        if backend == 'hf':
            from src.engine.inference import CPUEngine
            self.engine = CPUEngine(str(Path(root) / f'{slug}-hf'), max_seq_len=context, threads=threads, dtype='fp32')
            self.settings['precision'] = spec['hf_precision']
        elif backend == 'gguf':
            import llama_cpp
            if llama_cpp.__version__ != spec['llama_cpp_python']:
                raise ValueError(f"requires llama-cpp-python=={spec['llama_cpp_python']}")
            self.llm = llama_cpp.Llama(model_path=str(Path(root) / f'{slug}-gguf' / spec['gguf_file']),
                                       n_ctx=context, n_threads=threads, n_threads_batch=threads,
                                       n_batch=batch, n_ubatch=batch, n_gpu_layers=0, seed=0, verbose=False)
            trained = int(self.llm.metadata.get('qwen2.context_length', 0))
            if trained != spec['context_limit']:
                raise ValueError(f'GGUF context metadata disagrees with pinned specification: {trained}')
            self.settings.update(precision=spec['gguf_precision'],
                                 system_info=llama_cpp.llama_cpp.llama_print_system_info().decode())
        else:
            raise ValueError('backend must be hf or gguf')

    def check_prompt(self, case):
        ids = self.tok.encode(case['prompt'], add_special_tokens=False)
        if ids != case['input_ids']:
            raise ValueError('saved prompt does not match HF tokenization')
        if self.backend == 'gguf':
            actual = self.llm.tokenize(case['prompt'].encode('utf-8'), add_bos=False, special=True)
            if actual != ids:
                raise ValueError('HF/GGUF tokenizer mismatch; comparison would not be equivalent')

    def reset(self):
        if self.backend == 'hf':
            self.engine.reset_session('evaluation')
        else:
            self.llm.reset()

    def generate(self, ids, max_new=16, cold=True):
        if not ids or max_new <= 0 or len(ids) + max_new > self.context:
            raise ValueError('prompt plus output must fit the configured context')
        if cold:
            self.reset()
        def execute():
            if self.backend == 'hf':
                result = self.engine.generate(ids, session_id='evaluation', max_new_tokens=max_new,
                                              prompt_mode='full')
                result['output_ids'] = self.engine.sessions['evaluation'].input_ids[len(ids):]
                return result
            # Only reuse a verified complete cached prefix. A mismatch resets the
            # context; no estimated reuse count or implicit chat template is used.
            cached = list(self.llm.input_ids[:self.llm.n_tokens])
            reuse = len(cached) if cached and ids[:len(cached)] == cached and len(ids) > len(cached) else 0
            if not reuse:
                self.llm.reset()
            start = time.perf_counter()
            generated, arrivals = [], []
            stream = self.llm.generate(ids[reuse:], reset=False, temp=0, top_k=1, top_p=1,
                                       min_p=0, repeat_penalty=1)
            try:
                for token in stream:
                    generated.append(int(token))
                    arrivals.append(time.perf_counter())
                    if token == self.tok.eos_token_id or len(generated) >= max_new:
                        break
            finally:
                stream.close()
            return {'text': self.tok.decode(generated, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False),
                    'output_ids': generated, 'prompt_tokens': len(ids),
                    'generated_tokens': len(generated), 'reused_tokens': reuse,
                    'ttft_s': arrivals[0] - start,
                    'tpot_s': (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) if len(arrivals) > 1 else None,
                    'finish_reason': 'stop' if generated[-1] == self.tok.eos_token_id else 'length'}
        start = time.perf_counter()
        result, memory = measured(execute)
        return {**result, **memory, 'wall_s': time.perf_counter() - start,
                'evaluated_prompt_tokens': len(ids) - result['reused_tokens']}
