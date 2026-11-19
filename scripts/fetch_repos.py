#!/usr/bin/env python
"""Clone the pinned repository snapshots listed in configs/repos.yaml.

For each repository: clone (or reuse) into ``data/repos/<name>``, check out the pinned
commit in detached HEAD, verify HEAD matches, re-detect the licence and write
``data/repos/<name>.provenance.json`` (URL, commit, licence, retrieval date).

    python scripts/fetch_repos.py                 # all repositories
    python scripts/fetch_repos.py schedule        # selected ones
    python scripts/fetch_repos.py --index         # also build data/indexes/<name>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repoguard.repository.indexer import detect_licence  # noqa: E402


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def fetch(entry: dict, repos_dir: Path) -> dict:
    name, url, commit = entry["name"], entry["url"], entry["commit"]
    dest = repos_dir / name
    if not (dest / ".git").exists():
        print(f"[{name}] cloning {url}")
        git("clone", "--quiet", url, str(dest))
    else:
        print(f"[{name}] reusing {dest}")
    try:
        git("cat-file", "-e", f"{commit}^{{commit}}", cwd=dest)
    except subprocess.CalledProcessError:
        git("fetch", "--quiet", "origin", commit, cwd=dest)
    git("-c", "advice.detachedHead=false", "checkout", "--quiet", commit, cwd=dest)
    head = git("rev-parse", "HEAD", cwd=dest)
    if head != commit:
        raise SystemExit(f"[{name}] HEAD {head} != pinned {commit}")

    detected = detect_licence(dest)
    if detected != entry.get("licence"):
        print(
            f"[{name}] WARNING: detected licence {detected!r} != configured {entry.get('licence')!r}; check by hand"
        )
    provenance = {
        "name": name,
        "url": url,
        "commit": commit,
        "commit_date": git("show", "-s", "--format=%cI", commit, cwd=dest),
        "licence": entry.get("licence"),
        "licence_detected": detected,
        "role": entry.get("role"),
        "retrieved_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "local_path": str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest),
    }
    (repos_dir / f"{name}.provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[{name}] at {commit[:12]} ({provenance['commit_date']}), licence {detected}")
    return provenance


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("names", nargs="*", help="repository names (default: all)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "repos.yaml")
    ap.add_argument("--repos-dir", type=Path, default=ROOT / "data" / "repos")
    ap.add_argument("--index", action="store_true", help="build data/indexes/<name> after fetching")
    ap.add_argument("--index-config", type=Path, default=ROOT / "configs" / "indexer_config.yaml")
    args = ap.parse_args(argv)

    entries = yaml.safe_load(args.config.read_text(encoding="utf-8"))["repositories"]
    if args.names:
        unknown = set(args.names) - {e["name"] for e in entries}
        if unknown:
            ap.error(f"unknown repositories: {', '.join(sorted(unknown))}")
        entries = [e for e in entries if e["name"] in args.names]
    args.repos_dir.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        prov = fetch(entry, args.repos_dir)
        if args.index:
            from build_index import load_yaml_config

            from repoguard.repository import build_index

            cfg = load_yaml_config(args.index_config) if args.index_config.exists() else {}
            cfg["repository"] = {
                "name": prov["name"], "source_url": prov["url"],
                "commit": prov["commit"], "licence": prov["licence"],
            }  # fmt: skip
            art = build_index(
                args.repos_dir / entry["name"], ROOT / "data" / "indexes" / entry["name"], cfg
            )
            print(
                f"[{entry['name']}] indexed: {len(art.symbols)} symbols, {len(art.call_edges)} edges, {len(art.errors)} issues"
            )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
