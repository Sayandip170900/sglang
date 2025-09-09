"""Launch the inference server."""

import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.model_loader.prefetch import early_prefetch, wait_for_prefetch


if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    early_prefetch(
        model_path=server_args.model_path,
        served_model_name=server_args.served_model_name,
        revision=server_args.revision,
        spec_algo=server_args.speculative_algorithm,
        spec_draft_model_path=server_args.speculative_draft_model_path,
        tp_size=getattr(server_args, "tp_size", getattr(server_args, "tp", None)),
    )

    timeout = float(os.getenv("SGLANG_PREFETCH_TIMEOUT_SEC", "21600"))
    ok = wait_for_prefetch(
        model_path=server_args.model_path,
        served_model_name=server_args.served_model_name,
        revision=server_args.revision,
        timeout=timeout,
    )
    if not ok:
        raise RuntimeError("Prefetch failed or timed out; refusing to start")

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
