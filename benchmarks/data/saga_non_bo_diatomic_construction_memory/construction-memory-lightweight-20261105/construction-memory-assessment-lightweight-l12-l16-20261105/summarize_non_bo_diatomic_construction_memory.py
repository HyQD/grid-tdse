"""Summarize two-radial construction-memory JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEMA = "grid-tdse-two-radial-construction-memory-v1"


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    return parser.parse_args()


def _result_paths(paths):
    result = []
    for path in paths:
        if path.is_dir():
            result.extend(sorted(path.glob("*.json")))
        else:
            result.append(path)
    return result


def _stage(stages, name):
    return next(item for item in stages if item["stage"] == name)


def _mib(value):
    return value / 2**20


def main():
    arguments = _arguments()
    records = []
    for path in _result_paths(arguments.paths):
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        if result.get("schema") != SCHEMA:
            raise ValueError(f"unexpected schema in {path}")
        ranks = result["ranks"]
        baseline = [
            _stage(rank["stages"], "imports_complete")["current_rss_mib"]
            for rank in ranks
        ]
        final = [
            _stage(rank["stages"], "distributed_hamiltonian_built")["current_rss_mib"]
            for rank in ranks
        ]
        peak = [
            _stage(rank["stages"], "distributed_hamiltonian_built")["peak_rss_mib"]
            for rank in ranks
        ]
        categories = ranks[0]["array_inventory"]["unique_bytes_by_category"]
        unused = categories.get("expanded_pair_operators", 0) + categories.get(
            "dense_one_body", 0
        )
        state = max(rank["owned_complex_state_bytes"] for rank in ranks)
        records.append(
            {
                "profile": result["problem"]["profile_name"],
                "l_max": result["problem"]["l_max"],
                "layout_ranks": result["execution"]["layout_ranks"],
                "radial": "x".join(
                    str(value) for value in result["problem"]["radial_points"]
                ),
                "backend": result["backend"]["resolved"],
                "rss_growth": max(
                    value - base
                    for value, base in zip(final, baseline, strict=True)
                ),
                "peak": max(peak),
                "arrays": _mib(ranks[0]["array_inventory"]["unique_bytes_total"]),
                "unused": _mib(unused),
                "state": _mib(state),
                "ratio": unused / state,
                "path": path,
            }
        )

    records.sort(
        key=lambda item: (
            item["profile"],
            item["l_max"],
            item["layout_ranks"],
        )
    )
    print(
        f"{'profile':12s} {'lmax':>4s} {'layout':>6s} {'radial':>11s} "
        f"{'backend':>8s} {'RSS growth':>11s} {'peak':>9s} {'arrays':>9s} "
        f"{'unused':>9s} {'state':>9s} {'unused/state':>12s}"
    )
    for item in records:
        print(
            f"{item['profile']:12s} {item['l_max']:4d} {item['layout_ranks']:6d} "
            f"{item['radial']:>11s} {item['backend']:>8s} "
            f"{item['rss_growth']:9.1f}M {item['peak']:7.1f}M "
            f"{item['arrays']:7.1f}M {item['unused']:7.1f}M "
            f"{item['state']:7.1f}M {item['ratio']:12.1f}"
        )
    if any(item["backend"] != "py3nj" for item in records):
        print("\nWARNING: at least one run did not resolve the Gaunt backend to py3nj.")


if __name__ == "__main__":
    main()
