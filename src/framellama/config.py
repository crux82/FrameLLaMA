"""Small helpers shared by the scripts: repository paths and YAML configs.

Relative paths written inside a config file are resolved against the
repository root, so every command works no matter which directory it is run
from. Relative paths given on the command line are resolved against the
current directory, as usual.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(path):
    """Resolve a path from a config file (relative to the repository root)."""
    if path is None:
        return None
    path = Path(os.path.expanduser(str(path)))
    return path if path.is_absolute() else (REPO_ROOT / path)


def cli_path(path):
    """Resolve a path given on the command line (relative to the current directory)."""
    if path is None:
        return None
    return Path(os.path.expanduser(str(path))).resolve()


def load_config(path):
    """Load a YAML config. ``path`` may be given relative to the cwd or the repo root."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required: pip install pyyaml") from exc
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def section(config, name):
    """Return a config section, or an empty dict."""
    return dict(config.get(name) or {})


def require_file(path, what, hint=""):
    """Exit-friendly check that an input file exists."""
    path = Path(path)
    if not path.exists():
        msg = f"{what} not found: {path}"
        if hint:
            msg += f"\n{hint}"
        raise FileNotFoundError(msg)
    return path


# ---------------------------------------------------------------
# Data files that may live on Hugging Face
# ---------------------------------------------------------------
REMOTE_PREFIXES = ("hf://", "http://", "https://")


def data_source(spec, from_cli, what="Data file"):
    """Return a remote path/URL unchanged, or an existing local file as a string.

    Remote: hf://datasets/<org>/<repo>/<file> or an https URL (for example
    https://huggingface.co/datasets/<org>/<repo>/resolve/main/<file>).
    """
    spec = str(spec)
    if spec.startswith(REMOTE_PREFIXES):
        return spec
    path = cli_path(spec) if from_cli else repo_path(spec)
    require_file(path, what)
    return str(path)


def load_json_data(source):
    """Load a JSON list or JSONL file (local, hf:// or URL) as a datasets.Dataset."""
    import sys
    from datasets import load_dataset
    try:
        return load_dataset("json", data_files={"train": source}, split="train")
    except Exception as exc:
        text = str(exc)
        if str(source).startswith(REMOTE_PREFIXES) and any(
                k in text for k in ("401", "403", "Unauthorized", "Repository Not Found", "gated", "authentic")):
            sys.exit("ERROR: could not access the data on Hugging Face. If the dataset is private, "
                     "log in with an account that has access: huggingface-cli login\n"
                     f"Details: {exc}")
        raise


# ---------------------------------------------------------------
# Model selection shared by the evaluation scripts
# ---------------------------------------------------------------
def add_model_args(parser):
    """--adapted / --baseline (the two models of the paper) and --models (any other)."""
    g = parser.add_argument_group("models (default: --baseline and --adapted)")
    g.add_argument("--adapted", action="store_true",
                   help="Evaluate FrameLLaMA (evaluation.models.adapted in the config)")
    g.add_argument("--baseline", action="store_true",
                   help="Evaluate the base model Llama-3.1-8B-Instruct (evaluation.models.baseline)")
    g.add_argument("--models", nargs="+", default=None,
                   help="Other models: an alias from evaluation.models, a Hugging Face id or a local folder")


def select_models(args, ev):
    """Return [(name, model_id)] to evaluate. name is used in the output file names."""
    import re
    aliases = ev.get("models", {})
    keys = [k for k, on in (("baseline", args.baseline), ("adapted", args.adapted)) if on]
    keys += list(args.models or [])
    if not keys:
        keys = list(ev.get("run_models", ["baseline", "adapted"]))
    out = []
    for key in keys:
        model = aliases.get(key, key)
        name = key if key in aliases else re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("_")
        out.append((name, model))
    return out
