# /// script
# dependencies = ["gguf"]
# ///
"""Show which precision each tensor of a GGUF file is stored in, and the real bits per weight.

Works on any GGUF file and reads only its tensor list, not the weights. Prints a table by
storage type (Q4_K, Q6_K, F32 and so on): how many parameters use it, their share of the model,
and the bits per weight, plus the average for the whole file. With --tensors it also lists
which tensors got which type, which shows where a mixed format such as Q4_K_M spends its
extra bits.

Run from anywhere (uv installs the gguf package for this script by itself):
    uv run scripts/inspect_gguf.py path/to/model.gguf
    uv run scripts/inspect_gguf.py path/to/model.gguf --tensors
"""

import argparse
import re
from collections import defaultdict


def summarize(tensors):
    """Group tensors by storage type.

    `tensors` is an iterable of (name, shape, type_name, n_bytes). Returns
    {type_name: {"count": tensors, "params": parameters, "bytes": bytes}}.
    """
    by_type = defaultdict(lambda: {"count": 0, "params": 0, "bytes": 0})
    for _name, shape, type_name, n_bytes in tensors:
        params = 1
        for dim in shape:
            params *= int(dim)
        entry = by_type[type_name]
        entry["count"] += 1
        entry["params"] += params
        entry["bytes"] += int(n_bytes)
    return dict(by_type)


def by_tensor_name(tensors):
    """Group tensors by name, with the layer number replaced by N, then by storage type.

    Returns {name_pattern: {type_name: parameters}}.
    """
    groups = defaultdict(lambda: defaultdict(int))
    for name, shape, type_name, _n_bytes in tensors:
        params = 1
        for dim in shape:
            params *= int(dim)
        groups[re.sub(r"\.\d+\.", ".N.", name)][type_name] += params
    return {name: dict(types) for name, types in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("gguf_file")
    parser.add_argument("--tensors", action="store_true", help="also list the type of each tensor")
    parser.add_argument(
        "--min-millions",
        type=float,
        default=30,
        help="with --tensors, hide tensors smaller than this",
    )
    args = parser.parse_args()

    from gguf import GGUFReader  # imported here so the helpers above need no extra package

    reader = GGUFReader(args.gguf_file)
    tensors = [(t.name, t.shape, t.tensor_type.name, t.n_bytes) for t in reader.tensors]
    by_type = summarize(tensors)
    total_params = sum(v["params"] for v in by_type.values())
    total_bytes = sum(v["bytes"] for v in by_type.values())

    print(
        f"{len(tensors)} tensors, {total_params / 1e9:.3f}B parameters, {total_bytes / 1e9:.3f} GB"
    )
    print(f"average {total_bytes * 8 / total_params:.3f} bits per weight\n")
    print(f"{'type':8} {'tensors':>7} {'parameters':>12} {'share':>7} {'bits per weight':>16}")
    for name, v in sorted(by_type.items(), key=lambda kv: -kv[1]["params"]):
        bits = v["bytes"] * 8 / v["params"]
        print(
            f"{name:8} {v['count']:>7} {v['params'] / 1e9:>11.3f}B "
            f"{v['params'] / total_params:>7.1%} {bits:>16.3f}"
        )

    if args.tensors:
        print("\nwhich tensors got which type (parameters in millions):")
        groups = by_tensor_name(tensors)
        for name, types in sorted(groups.items(), key=lambda kv: -sum(kv[1].values())):
            if sum(types.values()) < args.min_millions * 1e6:
                continue
            print(f"  {name:36} " + ", ".join(f"{t} {p / 1e6:.0f}M" for t, p in types.items()))


if __name__ == "__main__":
    main()
