# this file incorporates code from Reiss et al. FACTOR(https://github.com/talreiss/FACTOR)

import argparse
import os

import librosa
import csv
import numpy as np
import torch
import torch.nn.functional as F
from python_speech_features import logfbank
from tqdm import tqdm

# Fix deprecation in numpy
np.float = np.float64
np.int = np.int_

# Ensure avhubert is importable as a package (needed for relative imports inside it)
import sys as _sys
import importlib as _importlib
_av_hubert_parent = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'av_hubert'))
if _av_hubert_parent not in _sys.path:
    _sys.path.insert(0, _av_hubert_parent)
# Register avhubert modules with fairseq via package imports (keeps relative imports working)
_importlib.import_module('avhubert.hubert_pretraining')
_importlib.import_module('avhubert.hubert')
_importlib.import_module('avhubert.hubert_asr')
# Explicitly load the avhubert utils submodule (avhubert.__init__ star-imports shadow 'utils')
avhubert_utils = _importlib.import_module('avhubert.utils')
from fairseq import checkpoint_utils

FPS = 25

def load_model(ckpt_path):
    models, _, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]
    if hasattr(model, "decoder"):
        print("Checkpoint: fine-tuned")
        model = model.encoder.w2v_model
    else:
        print("Checkpoint: pre-trained w/o fine-tuning")
    model.cuda().eval()
    return model, task

def load_transforms(task):
    return avhubert_utils.Compose([
        avhubert_utils.Normalize(0.0, 255.0),
        avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)
    ])

def compute_starting_silence(audio_path, threshold=0.0005, sr=16000):
    # compute the starting silence in seconds
    audio, _ = librosa.load(audio_path, sr=sr)
    for i, sample in enumerate(audio):
        if abs(sample) > threshold:
            return i / sr
    return len(audio) / sr


def load_audio(path, silence_duration=0, sample_rate=16000, stack_order_audio=4):
    wav_data, sr = librosa.load(path, sr=sample_rate)
    assert sr == sample_rate and len(wav_data.shape) == 1

    skiped_frames = int(silence_duration * FPS) * 640
    if silence_duration > 0:
        skiped_frames += 640
    wav_data = wav_data[skiped_frames:]

    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)

    if len(audio_feats) % stack_order_audio != 0:
        pad = stack_order_audio - len(audio_feats) % stack_order_audio
        audio_feats = np.concatenate([audio_feats, np.zeros((pad, audio_feats.shape[1]), dtype=audio_feats.dtype)])

    audio_feats = audio_feats.reshape(-1, stack_order_audio * audio_feats.shape[1])
    audio_feats = torch.from_numpy(audio_feats.astype(np.float32))
    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
    return audio_feats

def extract_features(model, video_path, audio_path, transform, trimmed):
    frames = avhubert_utils.load_video(video_path)
    frames = transform(frames)
    frames = torch.FloatTensor(frames).unsqueeze(0).unsqueeze(0).cuda()


    audio_silence = compute_starting_silence(audio_path) if trimmed else 0
    audio = load_audio(audio_path, silence_duration=audio_silence)[None, :, :].transpose(1, 2).cuda()

    skip_frames = int(audio_silence * FPS) + 1 if audio_silence > 0 else 0
    print(trimmed, skip_frames)
    frames = frames[:, :, skip_frames:]

    min_len = min(frames.shape[2], audio.shape[-1])
    frames, audio = frames[:, :, :min_len], audio[:, :, :min_len]

    with torch.no_grad():
        f_audio, _ = model.extract_finetune({"video": None, "audio": audio}, None, None)
        f_video, _ = model.extract_finetune({"video": frames, "audio": None}, None, None)
        f_mm, _ = model.extract_finetune({"video": frames, "audio": audio}, None, None)

    return f_audio.squeeze(0).cpu().numpy(), f_video.squeeze(0).cpu().numpy(), f_mm.squeeze(0).cpu().numpy()


def process_av1m(args, model, transform):
    file_paths = set()
    with open(args.metadata, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            file_paths.add(row["path"])

    for _, file_path in enumerate(tqdm(file_paths)):
        mouth_roi_path = os.path.join(args.data_path, file_path[:-4] + "_roi.mp4")
        audio_path = os.path.join(args.data_path, file_path[:-4] + ".wav")

        try:
            feature_audio, feature_vid, feature_multimodal = extract_features(model, mouth_roi_path, audio_path, transform, args.trimmed)
        except:
            print(f"Unprocessed for file: {mouth_roi_path}")
            continue

        save_dict = {
            "visual": feature_vid,
            "audio": feature_audio,
            "multimodal": feature_multimodal,
        }
        save_path = os.path.join(args.save_path, file_path.replace(".mp4", ".npz"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        np.savez(save_path, **save_dict)

def process_fakeavceleb(args, model, transform, category):
    file_paths = set()

    # Load metadata CSV and filter by category
    with open(args.metadata, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == category:
                path = os.path.join(row["path"].replace("FakeAVCeleb/", ""), row["filename"])
                file_paths.add(path)

    for _, file_path in enumerate(tqdm(file_paths)):
        mouth_roi_path = args.data_path + file_path[:-4] + "_roi.mp4"
        audio_path = args.data_path + file_path[:-4] + ".wav"

        try:
            feature_audio, feature_vid, feature_multimodal = extract_features(model, mouth_roi_path, audio_path, transform, args.trimmed)
        except Exception as e:
            print(f"Unprocessed for file: {mouth_roi_path}; error {e}")
            continue

        save_dict = {
            "visual": feature_vid,
            "audio": feature_audio,
            "multimodal": feature_multimodal,
        }
        save_path = os.path.join(args.save_path, file_path.replace(".mp4", ".npz"))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        np.savez(save_path, **save_dict)
        

def process_avlips(args, model, transform):
    """Extract AV-HuBERT features for AVLips 0_real / 1_fake directories.

    Expects:
      args.data_path   — root of avlips_preprocessed/ (contains 0_real/ and 1_fake/)
      args.wav_path    — root of wav directory (contains 0_real/ and 1_fake/)
      args.save_path   — output root (creates 0_real/ and 1_fake/ subdirs)
    """
    classes = ["0_real", "1_fake"]
    for cls in classes:
        roi_dir = os.path.join(args.data_path, cls)
        wav_dir = os.path.join(args.wav_path, cls)
        out_dir = os.path.join(args.save_path, cls)
        os.makedirs(out_dir, exist_ok=True)

        roi_files = sorted([f for f in os.listdir(roi_dir) if f.endswith("_roi.mp4")])
        print(f"\n[AVLips] Feature extraction for {cls}: {len(roi_files)} clips", flush=True)

        for roi_fname in tqdm(roi_files, desc=f"AVLips {cls}"):
            stem = roi_fname[:-len("_roi.mp4")]  # e.g. "0", "1000", ...
            roi_path = os.path.join(roi_dir, roi_fname)
            wav_path = os.path.join(wav_dir, stem + ".wav")
            npz_path = os.path.join(out_dir, stem + ".npz")

            if os.path.exists(npz_path):
                continue  # already done

            if not os.path.exists(wav_path):
                print(f"[WARN] WAV missing for {roi_fname}: {wav_path}")
                continue

            try:
                feature_audio, feature_vid, feature_multimodal = extract_features(
                    model, roi_path, wav_path, transform, args.trimmed
                )
            except Exception as e:
                print(f"[WARN] Failed for {roi_fname}: {e}")
                continue

            np.savez(npz_path,
                     visual=feature_vid,
                     audio=feature_audio,
                     multimodal=feature_multimodal)


def main():
    parser = argparse.ArgumentParser(description="Extract AVHubert features")
    parser.add_argument("--dataset", type=str, default="AV1M", help="Dataset to extract features for")
    parser.add_argument("--metadata", type=str,default="av1m_metadata/train_metadata.csv", help="Path to the dataset metadata (for AV1M this dictates the train/val/test split to extract features for)")
    parser.add_argument("--split", default="train", help="For AV1M: data split to process (e.g., val, train)")
    parser.add_argument("--ckpt_path", type=str, default="self_large_vox_433h.pt", help="Path to AVHubert checkpoint")
    parser.add_argument("--data_path", type=str, default="av1m_preprocessed/", help="Path to the root folder of preprocessed data (ROI videos)")
    parser.add_argument("--wav_path", type=str, default="", help="For AVLips: path to wav root dir (with 0_real/ and 1_fake/ subdirs)")
    parser.add_argument("--save_path", type=str, default="av1m_features/", help="Output directory for saving features")
    parser.add_argument("--category", choices=["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio", "all"], default="all", help="For FakeAVCeleb: select category (RealVideo-RealAudio, etc.)")
    parser.add_argument("--trimmed", action="store_true", help="Wether to trimmed to starting silence or not")
    args = parser.parse_args()

    # model
    model, task = load_model(args.ckpt_path)
    transform = load_transforms(task)

    if args.dataset == "AV1M":
        if args.split == "test":
            args.data_path = os.path.join(args.data_path, "val")
            args.save_path =  os.path.join(args.save_path, "val")
        else:
            args.data_path = os.path.join(args.data_path, "train")
            args.save_path =  os.path.join(args.save_path, "train")
        process_av1m(args, model, transform)
        
    elif args.dataset == "FakeAVCeleb":
        if args.category == "all":
            categories = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]
        elif args.category:
            categories = [args.category]

        for category in categories:
            process_fakeavceleb(args, model, transform, category)

    elif args.dataset == "AVLips":
        if not args.wav_path:
            parser.error("--wav_path is required for --dataset AVLips")
        process_avlips(args, model, transform)

if __name__ == "__main__":
    main()
