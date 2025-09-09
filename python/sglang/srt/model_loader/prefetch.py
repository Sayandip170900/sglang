import errno
import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

_prefetch_status: Dict[str, str] = {}
_prefetch_threads: Dict[str, threading.Thread] = {}

def _key(repo_id: str, revision: Optional[str]) -> str:
    return f"{repo_id}:{revision or 'main'}"

def _repo_id_snapshot(p: Path) -> Optional[str]:
    parts = list(p.resolve().parts)
    try:
        i = parts.index("hub")
        models_dir = parts[i + 1]
        if models_dir.startswith("models--"):
            _, org, name = models_dir.split("--", 2)
            return f"{org}/{name}"
    except Exception:
        pass
    return None

def _rev_snapshot(p: Path) -> Optional[str]:
    parts = list(p.resolve().parts)
    for i, x in enumerate(parts):
        if x == "snapshots" and i + 1 < len(parts):
            return parts[i + 1]
    return None

def _hf_cache_dir() -> str:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE as _HF_CACHE
    except Exception:
        _HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
    return (
        os.getenv("HF_HOME")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or _HF_CACHE
    )

def _has_tokenizer(d: Path) -> bool:
    tok_ok = any(
        (d / name).exists()
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "ice_text.model",
        )
    )
    cfg_ok = any((d / name).exists() for name in ("config.json", "config.yaml"))
    return tok_ok and cfg_ok

def _has_index_and_shards(d: Path) -> bool:
    idx = (d / "model.safetensors.index.json")
    if idx.exists():
        try:
            j = json.loads(idx.read_text())
            shards = {Path(s).name for s in j.get("weight_map", {}).values()}
            return bool(shards) and all((d / s).exists() for s in shards)
        except Exception:
            return False

    safes = list(d.glob("*.safetensors"))
    bins  = list(d.glob("*.bin"))
    if len(safes) > 1 or len(bins) > 1:
        return False

    for c in ("model.safetensors", "pytorch_model.safetensors", "model.bin", "pytorch_model.bin"):
        if (d / c).exists():
            return True
    return (len(safes) == 1) or (len(bins) == 1)

def _has_local_snapshot(p: Optional[str]) -> bool:
    if not p:
        return False
    d = Path(p)
    if not d.is_dir():
        return False
    ok = _has_index_and_shards(d) and _has_tokenizer(d)
    if ok:
        print(f"[prefetch] skip: local snapshot looks complete at {d}", flush=True)
    return ok

def _resolve_repo_rev_localdir(
    model_path: Optional[str],
    served_model_name: Optional[str],
    cli_revision: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if model_path:
        p = Path(model_path)
        if p.exists():
            rid = _repo_id_snapshot(p)
            rev = _rev_snapshot(p) if cli_revision is None else cli_revision
            return (rid, rev, str(p))
        if "/" in model_path:
            return (model_path, cli_revision, None)

    if served_model_name:
        return (served_model_name, cli_revision, None)

    return (None, cli_revision, None)

def _get_world_and_node_rank(tp_size: Optional[int]) -> Tuple[int, int]:
    def _int_env(*names, default=None):
        for n in names:
            v = os.getenv(n)
            if v and v.strip() and v.strip().lower() != "none":
                try:
                    return int(v)
                except ValueError:
                    pass
        return default

    world_rank = _int_env("RANK", "SLURM_PROCID", "PMI_RANK", default=0)
    node_rank_env = _int_env("NODE_RANK", default=None)
    if node_rank_env is not None:
        return world_rank, node_rank_env

    if tp_size and tp_size > 0:
        return world_rank, world_rank // tp_size

    return world_rank, 0

def _allow_patterns_default() -> Tuple[str, ...]:
    return (
        "*.safetensors",
        "*.bin",
        "*.gguf",
        "model.safetensors.index.json",
        "config.json",
        "config.yaml",
        "generation_config.json",
        "tokenizer*",
        "*.model",
        "*.txt",
        "special_tokens_map.json",
        "*.py",
        "*.md",
        "*.vocab",
        "*.merges",
        "*.tiktoken",
    )

def _snap_dl(
    repo_id: str,
    revision: Optional[str],
    cache_dir: str,
    allow_patterns: Optional[Tuple[str, ...]],
    local_dir_for_dl: Optional[str],
):
    k = _key(repo_id, revision)
    try:
        print(f"[prefetch] starting: {k}", flush=True)
        _prefetch_status[k] = "downloading"

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            resume_download=True,
            local_files_only=False,
            cache_dir=cache_dir,
            allow_patterns=list(allow_patterns or _allow_patterns_default()),
            local_dir=local_dir_for_dl,
            local_dir_use_symlinks=True,
        )
        _prefetch_status[k] = "complete"
        print(f"[prefetch] complete: {k}", flush=True)
    except Exception:
        _prefetch_status[k] = "failed"
        print(f"[prefetch] failed: {k}\n{traceback.format_exc()}", flush=True)

def _acquire_file_lock(lock_file: Path, retry: bool = True) -> bool:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, f"{os.getpid()} {int(time.time())}\n".encode())
        finally:
            os.close(fd)
        return True
    except OSError as e:
        if e.errno == errno.EEXIST and retry:
            try:
                if time.time() - lock_file.stat().st_mtime > 1800:
                    lock_file.unlink(missing_ok=True)
                    return _acquire_file_lock(lock_file, retry=False)
            except FileNotFoundError:
                pass
        return False

def _cleanup_dead_threads():
    global _prefetch_threads
    dead_keys = [k for k, t in _prefetch_threads.items() if not t.is_alive()]
    for k in dead_keys:
        del _prefetch_threads[k]

def _maybe_start_prefetch(
    repo_or_path: Optional[str],
    served_model_name: Optional[str],
    revision: Optional[str],
    tp_size: Optional[int],
    allow_patterns: Optional[Tuple[str, ...]],
):
    if _has_local_snapshot(repo_or_path):
        return

    repo_id, rev, local_dir_for_dl = _resolve_repo_rev_localdir(
        repo_or_path, served_model_name, revision
    )
    if not repo_id:
        print("[prefetch] skip: cannot resolve repo_id", flush=True)
        return

    lr = os.getenv("LOCAL_RANK")
    if lr and lr.strip().isdigit() and int(lr) != 0:
        print("[prefetch] skip: LOCAL_RANK != 0", flush=True)
        return

    if not lr:
        _, node_rank = _get_world_and_node_rank(tp_size)
        if node_rank != 0:
            print("[prefetch] skip: node_rank != 0", flush=True)
            return

    cache_dir = _hf_cache_dir()
    k = _key(repo_id, rev)

    _cleanup_dead_threads()

    if k in _prefetch_threads and _prefetch_threads[k].is_alive():
        print(f"[prefetch] skip: already running thread for {k}", flush=True)
        return
    if _prefetch_status.get(k) in ("downloading", "complete"):
        print(f"[prefetch] skip: status={_prefetch_status[k]} for {k}", flush=True)
        return

    locks_dir = Path(cache_dir) / "sglang_prefetch_locks"
    lock_file = locks_dir / f"{repo_id.replace('/', '--')}--{rev or 'main'}.lock"
    if not _acquire_file_lock(lock_file):
        print(f"[prefetch] skip: lock held for {k}", flush=True)
        return

    _prefetch_status[k] = "downloading"
    t = threading.Thread(
        target=_snap_dl,
        args=(repo_id, rev, cache_dir, allow_patterns, local_dir_for_dl),
        daemon=True,
        name=f"prefetch-{k}",
    )
    _prefetch_threads[k] = t
    t.start()

def early_prefetch(
    model_path: Optional[str],
    served_model_name: Optional[str],
    revision: Optional[str],
    spec_algo: Optional[str],
    spec_draft_model_path: Optional[str],
    tp_size: Optional[int] = None,
    allow_patterns: Optional[Tuple[str, ...]] = None,
):
    try:
        _maybe_start_prefetch(model_path, served_model_name, revision, tp_size, allow_patterns)
        if spec_algo and spec_algo.lower() == "eagle" and spec_draft_model_path:
            _maybe_start_prefetch(spec_draft_model_path, None, None, tp_size, allow_patterns)
    except Exception:
        print("[prefetch] early_prefetch failed:\n" + traceback.format_exc(), flush=True)

def wait_for_prefetch(
    model_path: Optional[str],
    served_model_name: Optional[str],
    revision: Optional[str],
    timeout: float = 300.0,
) -> bool:
    repo_id, rev, _ = _resolve_repo_rev_localdir(model_path, served_model_name, revision)
    if not repo_id:
        return True

    k = _key(repo_id, rev)
    if _prefetch_status.get(k) not in ("downloading", "complete", "failed"):
        if k in _prefetch_threads:
            _prefetch_status[k] = "downloading"
        else:
            return True

    start = time.time()
    while time.time() - start < timeout:
        st = _prefetch_status.get(k, "unknown")
        if st in ("complete", "failed"):
            return st == "complete"
        time.sleep(0.5)
    print(f"[prefetch] timeout waiting for {k}", flush=True)
    return False
