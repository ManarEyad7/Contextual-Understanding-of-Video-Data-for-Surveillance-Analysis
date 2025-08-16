import sys
import os

# Add the parent directory of 'LoVR' to sys.path, so it can find 'RAFT/core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RAFT')))

from tqdm import tqdm
import json
import cv2
import numpy as np
import torch
import argparse
from scipy.signal import savgol_filter, find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import RAFT from the core module
from core.raft import RAFT
from core.utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_tensor_rgb(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    t = torch.from_numpy(rgb).permute(2, 0, 1)[None]   # [1,3,H,W]
    return t.to(DEVICE)

@torch.no_grad()
def run_raft_video(args):
    # build model
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module.to(DEVICE).eval()

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open video: {args.video}"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or args.fps
    print(f"Input frames: {total} | Input FPS: {fps_in:.2f} | Using iters={args.iters}, max_side={args.max_side}")

    # optional downscale for speed
    def maybe_resize(img):
        if args.max_side <= 0:
            return img
        h, w = img.shape[:2]
        scale = args.max_side / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        return img

    ok, prev = cap.read()
    if not ok: raise RuntimeError("Empty/unsupported video")
    prev = maybe_resize(prev)
    motion = []        # per-frame motion energy (length N-1)
    idx_pairs = []     # (i,i+1) for which we computed flow

    pbar = tqdm(total=max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1, 0), desc='RAFT flow', unit='frame')
    while True:
        ok, curr = cap.read()
        if not ok: break
        curr = maybe_resize(curr)

        im1 = to_tensor_rgb(prev)
        im2 = to_tensor_rgb(curr)
        padder = InputPadder(im1.shape)
        im1p, im2p = padder.pad(im1, im2)

        _, flow_up = model(im1p, im2p, iters=args.iters, test_mode=True)  # [1,2,H,W]
        # motion energy = mean magnitude
        u, v = flow_up[:,0], flow_up[:,1]
        mag = torch.sqrt(u*u + v*v).mean().item()
        motion.append(mag)
        idx_pairs.append((len(motion)-1, len(motion)))  # (t, t+1) in original frame indexing

        prev = curr
        pbar.update(1)

    pbar.close()
    cap.release()
    return np.array(motion, dtype=np.float32), idx_pairs

def smooth_signal(x, window, poly):
    # Ensure odd window and >= poly+2
    window = max(window, poly+2) | 1
    return savgol_filter(x, window_length=window, polyorder=poly, mode='interp')

def detect_boundaries(motion_s, k, min_gap):
    # Use z-score threshold on smoothed signal + peak finding
    mu, sd = np.mean(motion_s), np.std(motion_s) + 1e-8
    height = mu + k*sd
    peaks, _ = find_peaks(motion_s, height=height, distance=min_gap)
    return peaks  # indices in motion space (between frames)

def segments_from_peaks(peaks, n_frames, min_len):
    # Convert peak indices (on transitions between t and t+1) into segments [start, end]
    cut_points = [int(p)+1 for p in peaks if 0 <= p < n_frames-1]
    cuts = sorted(set([0] + cut_points + [n_frames-1]))
    segs = []
    for s, e in zip(cuts[:-1], cuts[1:]):
        if e - s + 1 >= min_len:
            segs.append({"start_frame": int(s), "end_frame": int(e)})
    # If last tiny tail got dropped by min_len, merge it into previous
    if segs and segs[-1]["end_frame"] < n_frames-1:
        segs[-1]["end_frame"] = n_frames-1
    return segs

def have_ffmpeg():
    return os.system("ffmpeg -version >/dev/null 2>&1") == 0

def export_clip_with_fallback(start_time, end_time, out_path, video_path, fps_in):
    # Try to use libx264 first
    if have_ffmpeg():
        try:
            cmd = (
                f"ffmpeg -y -loglevel error -ss {start_time:.3f} -to {end_time:.3f} "
                f"-i \"{video_path}\" -c:v libx264 -pix_fmt yuv420p -an \"{out_path}\""
            )
            os.system(cmd)
        except Exception as e:
            print(f"Error with libx264 codec: {e}. Using MJPEG fallback...")
            fallback_to_mjpeg(start_time, end_time, out_path, video_path, fps_in)
    else:
        print("ffmpeg not found; using MJPEG fallback...")
        fallback_to_mjpeg(start_time, end_time, out_path, video_path, fps_in)

# Fallback to MJPEG
def fallback_to_mjpeg(start_time, end_time, out_path, video_path, fps_in):
    cmd = (
        f"ffmpeg -y -loglevel error -ss {start_time:.3f} -to {end_time:.3f} "
        f"-i \"{video_path}\" -c:v mjpeg -q:v 5 -an \"{out_path}\""
    )
    os.system(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="RAFT/models/raft-things.pth")
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--out_dir", default="mbts_out", help="output directory")
    ap.add_argument("--fps", type=int, default=24)  # fallback if video FPS is missing
    ap.add_argument("--iters", type=int, default=12, help="RAFT iters (20 for best, smaller for speed)")
    ap.add_argument("--max_side", type=int, default=640, help="downscale longest side to this (0=off)")
    # smoothing / detection
    ap.add_argument("--savgol_window", type=int, default=31)
    ap.add_argument("--savgol_poly", type=int, default=3)
    ap.add_argument("--z_k", type=float, default=2.0, help="k * std above mean")
    ap.add_argument("--min_gap", type=int, default=12, help="minimum frames between boundaries")
    ap.add_argument("--min_seg_len", type=int, default=24, help="minimum segment length (frames)")
    # extras
    ap.add_argument("--export_clips", action="store_true", help="export each segment as its own video")
    ap.add_argument("--side_by_side", action="store_true", help="export clips with original|flow overlay side-by-side")
    # RAFT flags kept for compatibility
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--mixed_precision", action="store_true")
    ap.add_argument("--alternate_corr", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) motion signal
    motion, idx_pairs = run_raft_video(args)
    n_frames = len(motion) + 1  # motion defined between consecutive frames

    # 2) smooth
    motion_s = smooth_signal(motion, args.savgol_window, args.savgol_poly)

    # 3) detect peaks (boundaries)
    peaks = detect_boundaries(motion_s, args.z_k, args.min_gap)

    # 4) build segments
    segs = segments_from_peaks(peaks, n_frames=n_frames, min_len=args.min_seg_len)

    # 5) save artifacts
    csv_path = os.path.join(args.out_dir, "motion.csv")
    with open(csv_path, "w") as f:
        f.write("t,motion_raw,motion_smooth,is_boundary\n")
        boundary_set = set(peaks.tolist())
        for t, (mr, ms) in enumerate(zip(motion, motion_s)):
            isb = 1 if t in boundary_set else 0
            f.write(f"{t},{mr:.6f},{ms:.6f},{isb}\n")

    seg_json = os.path.join(args.out_dir, "segments.json")
    with open(seg_json, "w") as f:
        json.dump({"segments": segs, "n_frames": n_frames}, f, indent=2)

    # 6) plot
    plt.figure(figsize=(12,4))
    plt.plot(motion, label="motion raw", alpha=0.4)
    plt.plot(motion_s, label="motion smoothed", linewidth=2)
    for p in peaks:
        plt.axvline(p, color="r", linestyle="--", alpha=0.6)
    plt.title("Motion Energy & Boundaries")
    plt.xlabel("transition index (between frames t and t+1)")
    plt.ylabel("mean flow magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "plot.png"), dpi=150)

    print(f"\nSaved:\n- {csv_path}\n- {seg_json}\n- {os.path.join(args.out_dir, 'plot.png')}")

    # 7) optional: export clips
    if args.export_clips and segs:
        clips_dir = os.path.join(args.out_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        # probe fps & frame size from the original video
        cap = cv2.VideoCapture(args.video)
        assert cap.isOpened(), f"Cannot open video: {args.video}"
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or args.fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Export clips using ffmpeg (or MJPEG if ffmpeg fails)
        for i, seg in enumerate(segs):
            s, e = seg["start_frame"], seg["end_frame"]
            start_time = s / fps_in
            end_time   = (e + 1) / fps_in  # inclusive segment end
            out_path = os.path.join(clips_dir, f"seg_{i:04d}_{s:06d}_{e:06d}.mp4")
            export_clip_with_fallback(start_time, end_time, out_path, args.video, fps_in)

if __name__ == "__main__":
    main()

    