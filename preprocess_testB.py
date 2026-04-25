"""
Preprocess a shard of testB raw MP4s → _roi.mp4 + .wav
Usage:
    python preprocess_testB.py --shard_idx 0 --num_shards 50
"""
import argparse
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add av_hubert so preparation.align_mouth is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AVHUBERT_PKG = os.path.join(_SCRIPT_DIR, "av_hubert", "avhubert")
sys.path.insert(0, _AVHUBERT_PKG)
sys.path.insert(0, os.path.dirname(_AVHUBERT_PKG))

# Reuse preprocess_video from deepfake_preprocess
sys.path.insert(0, _SCRIPT_DIR)
from deepfake_preprocess import preprocess_video

FACE_PREDICTOR_PATH = os.path.join(_SCRIPT_DIR, "content/data/misc/shape_predictor_68_face_landmarks.dat")
MEAN_FACE_PATH      = os.path.join(_SCRIPT_DIR, "content/data/misc/20words_mean_face.npy")

VIDEO_DIR = "/data/projects/punim2637/nnliang/Datasets/AVDeepfake1MPlusPlus/testB/videos/testB"
SAVE_DIR  = "/data/projects/punim2637/nnliang/Datasets/AVDeepfake1MPlusPlus/testB/preprocessed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_idx",  type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=50)
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_videos = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4"))
    shards = np.array_split(all_videos, args.num_shards)
    my_videos = shards[args.shard_idx].tolist()

    print(f"Shard {args.shard_idx}/{args.num_shards}: {len(my_videos)} videos", flush=True)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                preprocess_video,
                VIDEO_DIR,
                fname,
                SAVE_DIR,
                FACE_PREDICTOR_PATH,
                MEAN_FACE_PATH,
            ): fname
            for fname in my_videos
        }
        ok, fail = 0, 0
        for future in tqdm(as_completed(futures), total=len(futures)):
            fname = futures[future]
            try:
                result = future.result()
                if result:
                    ok += 1
                else:
                    fail += 1
                    print(f"[WARN] Failed: {fname}", flush=True)
            except Exception as e:
                fail += 1
                print(f"[ERROR] {fname}: {e}", flush=True)

    print(f"\nDone. OK={ok}, Failed={fail}", flush=True)


if __name__ == "__main__":
    main()
