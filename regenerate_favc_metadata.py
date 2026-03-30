#!/usr/bin/env python3
"""
Regenerate FakeAVCeleb metadata directly from the filesystem using the
canonical directory names.
"""
import os
import csv
from pathlib import Path


CORRECT_DIRS = {
    "Caucasian_American": "Caucasian (American)",
    "Caucasian_European": "Caucasian (European)",
    "Asian_East": "Asian (East)",
    "Asian_South": "Asian (South)",
    "African": "African"
}

def main():
    favc_root = "/data/scratch/projects/punim2637/nnliang/Datasets/FakeAVCeleb_v1.2"
    output_csv = "/data/projects/punim2637/nnliang/AVH-Align/av1m_metadata/favc_metadata_corrected.csv"
    
    rows = []
    categories = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]
    
    for category in categories:
        cat_path = os.path.join(favc_root, category)
        if not os.path.exists(cat_path):
            continue
        
        # Walk all video files in the category subtree.
        for video_file in Path(cat_path).rglob("*.mp4"):
            rel_path = video_file.relative_to(favc_root)
            path_parts = list(rel_path.parts)
            
            # Build path: FakeAVCeleb/category/race/gender/id/
            full_dir = f"FakeAVCeleb/{'/'.join(path_parts[:-1])}/"
            filename = path_parts[-1]
            
            rows.append({
                "type": category,
                "path": full_dir,
                "filename": filename
            })
    
    print(f"Found {len(rows)} videos")
    print(f"By category:")
    from collections import Counter
    counts = Counter(row["type"] for row in rows)
    for cat, count in sorted(counts.items()):
        print(f"  {cat}: {count}")
    
    # Write the corrected metadata CSV.
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["type", "path", "filename"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nSaved to: {output_csv}")
    print(f"Total: {len(rows)} entries")

if __name__ == "__main__":
    main()
