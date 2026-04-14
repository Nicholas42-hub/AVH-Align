#!/usr/bin/env python3
"""
Test preprocessing on a single video locally - no SLURM needed.
This will show us exactly what's happening.
"""

import os
import sys
os.environ['PYTHONNOUSERSITE'] = '1'
os.environ['PATH'] = '/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/FFmpeg/4.4.2/bin:/data/projects/punim2637/nnliang/avh_env/bin:' + os.environ['PATH']
os.environ['LD_LIBRARY_PATH'] = '/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/FFmpeg/4.4.2/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Configure skvideo before importing deepfake_preprocess
import skvideo
skvideo.setFFmpegPath("/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/FFmpeg/4.4.2/bin")

# Add avhubert to path
sys.path.insert(0, '/data/projects/punim2637/nnliang/AVH-Align/av_hubert/avhubert')

from deepfake_preprocess import preprocess_video

# Paths
FACE_PREDICTOR = "/data/projects/punim2637/nnliang/AVH-Align/content/data/misc/shape_predictor_68_face_landmarks.dat"
MEAN_FACE = "/data/projects/punim2637/nnliang/AVH-Align/content/data/misc/20words_mean_face.npy"

# Pick one test video
input_dir = "/data/projects/punim2637/nnliang/Datasets/AVDeepfake1M/train_batch_extracted_job1/train"
test_video = "id00145/nGLEGeuxlbY/00122/real.mp4"  # A simple real video
output_dir = "/tmp/test_preprocess_output"

print("="*70)
print("SINGLE VIDEO PREPROCESSING TEST")
print("="*70)
print(f"Input directory: {input_dir}")
print(f"Test video: {test_video}")
print(f"Output directory: {output_dir}")
print()

# Check if input exists
full_input = os.path.join(input_dir, test_video)
print(f"Checking input video: {full_input}")
if os.path.exists(full_input):
    size_mb = os.path.getsize(full_input) / (1024**2)
    print(f"✓ Input exists ({size_mb:.2f} MB)")
else:
    print(f"✗ Input NOT found!")
    sys.exit(1)

print()
print("Starting preprocessing...")
print("-"*70)

try:
    result = preprocess_video(
        input_video_dir=input_dir,
        video_filename=test_video,
        output_video_dir=output_dir,
        face_predictor_path=FACE_PREDICTOR,
        mean_face_path=MEAN_FACE
    )
    
    print("-"*70)
    print(f"\nPreprocessing returned: {result}")
    
    if result:
        # Check outputs
        expected_roi = os.path.join(output_dir, test_video[:-4] + '_roi.mp4')
        expected_wav = os.path.join(output_dir, test_video[:-4] + '.wav')
        
        print("\nChecking outputs:")
        if os.path.exists(expected_roi):
            size_mb = os.path.getsize(expected_roi) / (1024**2)
            print(f"✓ ROI video created: {expected_roi} ({size_mb:.2f} MB)")
        else:
            print(f"✗ ROI video NOT created: {expected_roi}")
            
        if os.path.exists(expected_wav):
            size_mb = os.path.getsize(expected_wav) / (1024**2)
            print(f"✓ Audio created: {expected_wav} ({size_mb:.2f} MB)")
        else:
            print(f"✗ Audio NOT created: {expected_wav}")
            
        # Show directory structure
        print(f"\nOutput directory structure:")
        os.system(f"find {output_dir} -type f | head -10")
        
        if os.path.exists(expected_roi) and os.path.exists(expected_wav):
            print("\n" + "="*70)
            print("✓ SUCCESS! Preprocessing works!")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("✗ FAILED! Files not created despite returning True")
            print("="*70)
    else:
        print("\n✗ Preprocessing returned False")
        
except Exception as e:
    print("-"*70)
    print(f"\n✗ Exception occurred: {e}")
    import traceback
    traceback.print_exc()
