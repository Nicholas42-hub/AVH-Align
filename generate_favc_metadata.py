"""
Generate FakeAVCeleb metadata CSV in the format expected by
deepfake_preprocess.py and deepfake_feature_extraction.py.

Input:  avh_sup/csv_metadata/favc/{train,val,test}_split.csv
        (columns: source, category, full_path)
Output: av1m_metadata/favc_metadata.csv
        (columns: type, path, filename)

Usage:
    python generate_favc_metadata.py               # all splits
    python generate_favc_metadata.py --split test  # test only
"""

import argparse
import os
import pandas as pd

SPLITS = ["train", "val", "test"]

CATEGORY_MAP = {
    "A": "RealVideo-RealAudio",
    "B": "RealVideo-FakeAudio",
    "C": "FakeVideo-RealAudio",
    "D": "FakeVideo-FakeAudio",
}


def full_path_to_meta(full_path: str) -> dict:
    """
    'FakeAVCeleb/RealVideo-RealAudio/African/men/id00166/00010.mp4'
    → {'type': 'RealVideo-RealAudio',
       'path': 'FakeAVCeleb/RealVideo-RealAudio/African/men/id00166/',
       'filename': '00010.mp4'}
    """
    parts = full_path.split("/")  # ['FakeAVCeleb', 'RealVideo-RealAudio', ..., 'file.mp4']
    return {
        "type": parts[1],
        "path": "/".join(parts[:-1]) + "/",
        "filename": parts[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS + ["all"], default="all",
                        help="Which split(s) to include (default: all)")
    parser.add_argument(
        "--csv_dir",
        default="avh_sup/csv_metadata/favc",
        help="Directory containing train_split.csv / val_split.csv / test_split.csv",
    )
    parser.add_argument(
        "--out",
        default="av1m_metadata/favc_metadata.csv",
        help="Output metadata CSV path",
    )
    args = parser.parse_args()

    splits = SPLITS if args.split == "all" else [args.split]

    frames = []
    for split in splits:
        csv_path = os.path.join(args.csv_dir, f"{split}_split.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] {csv_path} not found — skipping")
            continue
        df = pd.read_csv(csv_path)
        frames.append(df)
        print(f"  {split} : {len(df)} rows")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Total   : {len(combined)} rows")

    rows = [full_path_to_meta(fp) for fp in combined["full_path"]]
    meta_df = pd.DataFrame(rows).drop_duplicates()
    print(f"Unique  : {len(meta_df)} rows")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta_df.to_csv(args.out, index=False)
    print(f"Saved → {args.out}")

    # Show type distribution
    print("\nType distribution:")
    print(meta_df["type"].value_counts().to_string())


if __name__ == "__main__":
    main()
