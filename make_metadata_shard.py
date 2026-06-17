#!/usr/bin/env python3
"""Create a row-contiguous metadata shard from a CSV file.

The shard index is zero-based. Rows are split as evenly as possible across the
requested number of shards while preserving the original CSV header.
"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV with header.")
    parser.add_argument("--output", required=True, help="Output shard CSV.")
    parser.add_argument("--shard-id", type=int, required=True, help="Zero-based shard id.")
    parser.add_argument("--num-shards", type=int, required=True, help="Total number of shards.")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")

    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise RuntimeError(f"No CSV header found in {args.input}")

    n = len(rows)
    begin = (n * args.shard_id) // args.num_shards
    end = (n * (args.shard_id + 1)) // args.num_shards
    shard_rows = rows[begin:end]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shard_rows)

    print(
        f"Wrote shard {args.shard_id}/{args.num_shards}: "
        f"rows [{begin}, {end}) = {len(shard_rows)} rows -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
