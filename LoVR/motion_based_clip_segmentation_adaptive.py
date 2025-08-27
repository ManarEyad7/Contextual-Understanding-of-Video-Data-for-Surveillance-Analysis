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
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import RAFT from the core module
from core.raft import RAFT
from core.utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_tensor_rgb(frame_bgr):
    """Convert OpenCV BGR image to RGB float tensor [1,3,H,W] on DEVICE."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    t = torch.from_numpy(rgb).permute(2, 0, 1)[None]   # [1,3,H,W]
    return t.to(DEVICE)


def have_ffmpeg():
    return os.system("ffmpeg -version >/dev/null 2>&1") == 0


def export_clip_with_fallback(start_time, end_time, out_path, video_path, fps_in):
    """Try H.264; fallback to MJPEG if needed."""
    if have_ffmpeg():
        try:
            cmd = (
                f"ffmpeg -y -loglevel error -ss {start_time:.3f} -to {end_time:.3f} "
                f"-i \"{video_path}\" -c:v libx264 -pix_fmt yuv420p -an \"{out_path}\""
            )
            rc = os.system(cmd)
            if rc != 0:
                raise RuntimeError("ffmpeg returned non-zero exit code")
        except Exception:
            fallback_to_mjpeg(start_time, end_time, out_path, video_path, fps_in)
    else:
        fallback_to_mjpeg(start_time, end_time, out_path, video_path, fps_in)


def fallback_to_mjpeg(start_time, end_time, out_path, video_path, fps_in):
    cmd = (
        f"ffmpeg -y -loglevel error -ss {start_time:.3f} -to {end_time:.3f} "
        f"-i \"{video_path}\" -c:v mjpeg -q:v 5 -an \"{out_path}\""
    )
    os.system(cmd)


@torch.no_grad()
def run_raft_video(args):
    # Build RAFT
    model = torch.nn.DataParallel(RAFT(args))
    state = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(state)
    model = model.module.to(DEVICE).eval()

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open video: {args.video}"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_meta = cap.get(cv2.CAP_PROP_FPS)
    fps_in = args.fps if args.force_fps else (fps_meta or args.fps)
    try:
        fps_meta_str = f"{fps_meta:.2f}"
    except Exception:
        fps_meta_str = "NA"

    print(f"Input frames: {total} | FPS(meta)={fps_meta_str} | Using FPS={fps_in:.2f} | "
          f"iters={args.iters}, max_side={args.max_side}")

    def maybe_resize(img):
        if args.max_side <= 0:
            return img
        h, w = img.shape[:2]
        scale = args.max_side / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        return img

    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Empty/unsupported video")
    prev = maybe_resize(prev)

    motion = []        # per-frame motion energy (length N-1)
    idx_pairs = []     # (t, t+1) indices

    pbar = tqdm(total=max(total-1, 0), desc='RAFT flow', unit='frame')
    while True:
        ok, curr = cap.read()
        if not ok:
            break
        curr = maybe_resize(curr)

        im1 = to_tensor_rgb(prev)
        im2 = to_tensor_rgb(curr)
        padder = InputPadder(im1.shape)
        im1p, im2p = padder.pad(im1, im2)

        _, flow_up = model(im1p, im2p, iters=args.iters, test_mode=True)  # [1,2,H,W]
        u = flow_up[:, 0]  # [1,H,W]
        v = flow_up[:, 1]

        # --- Global motion removal (optional) ---
        if args.remove_global_motion:
            med_u = torch.median(u)
            med_v = torch.median(v)
            u = u - med_u
            v = v - med_v

        # magnitude per pixel
        mag = torch.sqrt(u*u + v*v)  # [1,H,W]

        # --- Diagonal normalization (optional) ---
        _, H, W = u.shape
        if args.diag_norm:
            diag = float((H**2 + W**2) ** 0.5)
            mag = mag / diag

        vals = mag.reshape(-1)

        # --- Energy statistic over pixels ---
        if args.energy == "p90":
            energy = torch.quantile(vals, 0.90).item()
        elif args.energy == "p95":
            energy = torch.quantile(vals, 0.95).item()
        else:  # mean
            energy = vals.mean().item()

        motion.append(energy)
        idx_pairs.append((len(motion)-1, len(motion)))  # (t, t+1)

        prev = curr
        pbar.update(1)

    pbar.close()
    cap.release()
    return np.array(motion, dtype=np.float32), idx_pairs, float(fps_in)


def smooth_signal(x, window, poly):
    # Ensure odd window and >= poly+2
    window = max(window, poly + 2) | 1
    return savgol_filter(x, window_length=window, polyorder=poly, mode='interp')


def compute_thresholds(x, method, k_hi, k_lo, q_hi, q_lo, robust=False):
    if method == 'quantile':
        thr_hi = float(np.quantile(x, q_hi))
        thr_lo = float(np.quantile(x, q_lo))
    else:  # 'z'
        if robust:
            med = float(np.median(x))
            mad = float(1.4826 * np.median(np.abs(x - med)) + 1e-8)
            thr_hi = med + k_hi * mad
            thr_lo = med + k_lo * mad
        else:
            mu = float(np.mean(x))
            sd = float(np.std(x) + 1e-8)
            thr_hi = mu + k_hi * sd
            thr_lo = mu + k_lo * sd
    # Ensure thr_lo <= thr_hi
    if thr_lo > thr_hi:
        thr_lo = thr_hi - 1e-6
    return thr_hi, thr_lo


def hysteresis_mask(x, thr_hi, thr_lo):
    """Return boolean mask of 'active' indices using hysteresis gating."""
    active = np.zeros_like(x, dtype=bool)
    on = False
    for i, val in enumerate(x):
        if not on and val >= thr_hi:
            on = True
        elif on and val <= thr_lo:
            on = False
        active[i] = on
    return active


def rolling_hysteresis_mask(x, fps, win_sec=30, k_hi=1.2, k_lo=0.6):
    """
    Adaptive (rolling) hysteresis using robust median/MAD per window.
    Keeps sensitivity when baseline drifts over time.
    """
    n = len(x)
    win = max(3, int(win_sec * fps))
    on = False
    active = np.zeros(n, dtype=bool)
    for i in range(n):
        a = max(0, i - win // 2)
        b = min(n, i + win // 2)
        w = x[a:b]
        med = np.median(w)
        mad = 1.4826 * np.median(np.abs(w - med)) + 1e-8
        hi = med + k_hi * mad
        lo = med + k_lo * mad
        if not on and x[i] >= hi:
            on = True
        elif on and x[i] <= lo:
            on = False
        active[i] = on
    return active


def fill_small_gaps(mask, max_hole):
    """Fill False runs of length <= max_hole between True regions."""
    if max_hole <= 0:
        return mask
    m = mask.copy()
    n = len(m)
    i = 0
    while i < n:
        if m[i]:
            i += 1
            continue
        j = i
        while j < n and not m[j]:
            j += 1
        left_on = (i - 1 >= 0) and m[i - 1]
        right_on = (j < n) and m[j] if j < n else False
        if left_on and right_on and (j - i) <= max_hole:
            m[i:j] = True
        i = j
    return m


def enforce_min_on_off(mask, min_on, min_off):
    """Convert mask to segments (in motion-index space) enforcing min_on and min_off."""
    # First: extract raw active runs
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) >= min_on:
                runs.append((i, j - 1))  # inclusive
            i = j
        else:
            i += 1
    if not runs:
        return []

    # Then: merge runs separated by short off gaps (< min_off)
    merged = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        gap = s - pe - 1
        if gap < min_off:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def quality_filter(segs, x_smooth, method, k_seg, q_seg, min_active_ratio, min_mean=None):
    """Drop segments with weak sustained activity."""
    if not segs:
        return []
    if method == 'quantile' and q_seg is not None:
        seg_thr = float(np.quantile(x_smooth, q_seg))
    elif method == 'z' and k_seg is not None:
        mu, sd = float(np.mean(x_smooth)), float(np.std(x_smooth) + 1e-8)
        seg_thr = mu + k_seg * sd
    else:
        seg_thr = None

    kept = []
    for s, e in segs:
        w = x_smooth[s:e+1]
        ok = True
        if min_mean is not None and np.mean(w) < min_mean:
            ok = False
        if ok and seg_thr is not None and min_active_ratio > 0.0:
            ratio = float(np.mean(w >= seg_thr))
            if ratio < min_active_ratio:
                ok = False
        if ok:
            kept.append((s, e))
    return kept


def motion_segments_to_frame_segs(segs_motion, n_frames):
    """Convert motion-index segments to frame-index segments [start_frame, end_frame]."""
    out = []
    for s, e in segs_motion:
        start_f = int(s)            # motion at index s is between frames s and s+1
        end_f = int(e + 1)          # inclusive end frame
        start_f = max(0, min(start_f, n_frames - 1))
        end_f   = max(0, min(end_f,   n_frames - 1))
        if end_f >= start_f:
            out.append({"start_frame": start_f, "end_frame": end_f})
    return out


def pad_and_merge_frame_segs(segs, n_frames, pad_pre, pad_post):
    """
    Expand each [start_frame, end_frame] by pad_pre/post (frames), clamp to [0, n_frames-1],
    then merge overlaps/touches.
    """
    if not segs:
        return []
    expanded = []
    for s in segs:
        a = max(0, s["start_frame"] - pad_pre)
        b = min(n_frames - 1, s["end_frame"] + pad_post)
        expanded.append((a, b))
    # merge overlapping / touching ranges
    expanded.sort()
    merged = [list(expanded[0])]
    for a, b in expanded[1:]:
        if a <= merged[-1][1] + 1:            # overlap or touch
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [{"start_frame": int(a), "end_frame": int(b)} for a, b in merged]


def merge_by_gap_seconds(segs, fps, max_gap_sec=4.0):
    """Merge adjacent segments whose gap is <= max_gap_sec."""
    if not segs:
        return segs
    out = [segs[0]]
    for s in segs[1:]:
        gap = (s["start_frame"] - out[-1]["end_frame"] - 1) / fps
        if gap <= max_gap_sec:
            out[-1]["end_frame"] = max(out[-1]["end_frame"], s["end_frame"])
        else:
            out.append(s)
    return out


def segs_to_timestamps(segs, fps, include_end=True, ndigits=3, clamp_to=None):
    """Return [[start_sec, end_sec], ...]."""
    out = []
    for s in segs:
        a = s["start_frame"] / fps
        b = (s["end_frame"] + (1 if include_end else 0)) / fps  # inclusive end
        if clamp_to is not None:
            b = min(b, clamp_to)
        out.append([round(a, ndigits), round(b, ndigits)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="RAFT/models/raft-things.pth")
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--out_dir", default="mbts_out", help="output directory")
    ap.add_argument("--fps", type=int, default=24, help="fallback FPS if missing in metadata")
    ap.add_argument("--force_fps", action="store_true",
                    help="Ignore video metadata FPS and use --fps for all timing.")
    ap.add_argument("--iters", type=int, default=12, help="RAFT iters (20 for best, smaller for speed)")
    ap.add_argument("--max_side", type=int, default=640, help="downscale longest side to this (0=off)")

    # --- Motion options ---
    ap.add_argument("--remove_global_motion", action="store_true", help="subtract median flow each frame")
    ap.add_argument("--diag_norm", action="store_true", help="divide motion by image diagonal for scale invariance")
    ap.add_argument("--energy", choices=["mean", "p90", "p95"], default="mean",
                    help="statistic over per-pixel flow magnitude for motion energy")

    # --- Smoothing ---
    ap.add_argument("--savgol_window", type=int, default=31)
    ap.add_argument("--savgol_poly", type=int, default=3)

    # --- Thresholding (global) ---
    ap.add_argument("--thr", choices=["z", "quantile"], default="z",
                    help="use z-score (mean+k*std/median+MAD) or quantiles")
    ap.add_argument("--robust_thr", action="store_true",
                    help="with --thr z, use median/MAD instead of mean/std")
    ap.add_argument("--k_hi", type=float, default=2.0, help="high threshold: mean/median + k_hi*std/MAD (z-mode)")
    ap.add_argument("--k_lo", type=float, default=1.0, help="low threshold: mean/median + k_lo*std/MAD (z-mode)")
    ap.add_argument("--q_hi", type=float, default=0.85, help="high threshold quantile (quantile-mode)")
    ap.add_argument("--q_lo", type=float, default=0.60, help="low threshold quantile (quantile-mode)")

    # --- Adaptive (rolling) hysteresis ---
    ap.add_argument("--use_rolling", action="store_true", help="use adaptive rolling thresholds instead of global")
    ap.add_argument("--roll_win_sec", type=float, default=30.0, help="window (seconds) for rolling hysteresis")
    ap.add_argument("--roll_k_hi", type=float, default=1.2, help="hi multiplier for MAD in rolling hysteresis")
    ap.add_argument("--roll_k_lo", type=float, default=0.6, help="lo multiplier for MAD in rolling hysteresis")

    # --- Post-processing & quality ---
    ap.add_argument("--min_on", type=int, default=24, help="min active duration (frames)")
    ap.add_argument("--max_hole", type=int, default=6, help="fill quiet dips of up to this many frames")
    ap.add_argument("--min_off", type=int, default=24, help="min quiet duration between segments (frames)")
    ap.add_argument("--seg_k", type=float, default=1.0, help="segment threshold: mean + seg_k*std (z-mode)")
    ap.add_argument("--seg_q", type=float, default=0.70, help="segment threshold quantile (quantile-mode)")
    ap.add_argument("--min_active_ratio", type=float, default=0.20,
                    help="min %% of frames in segment above segment-threshold")
    ap.add_argument("--min_mean", type=float, default=None,
                    help="drop segments with mean(smoothed motion) < this absolute value")
    ap.add_argument("--pad_pre",  type=int, default=0, help="extra frames BEFORE each segment")
    ap.add_argument("--pad_post", type=int, default=0, help="extra frames AFTER each segment")
    ap.add_argument("--merge_gap_sec", type=float, default=4.0,
                    help="merge adjacent segments whose gap (seconds) <= this; 0 disables")

    # --- Outputs ---
    ap.add_argument("--export_clips", action="store_true", help="export each segment as its own video")
    ap.add_argument("--side_by_side", action="store_true", help="(reserved) export with overlays")
    # RAFT flags kept for compatibility
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--mixed_precision", action="store_true")
    ap.add_argument("--alternate_corr", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) motion signal
    motion_raw, idx_pairs, fps_in = run_raft_video(args)
    n_frames = len(motion_raw) + 1

    # 2) smooth
    motion_s = smooth_signal(motion_raw, args.savgol_window, args.savgol_poly)

    # 3) global thresholds (for plot & global mode)
    thr_hi, thr_lo = compute_thresholds(
        motion_s, method=args.thr,
        k_hi=args.k_hi, k_lo=args.k_lo,
        q_hi=args.q_hi, q_lo=args.q_lo,
        robust=args.robust_thr
    )

    # 4) hysteresis gating
    if args.use_rolling:
        active = rolling_hysteresis_mask(
            motion_s, fps_in, win_sec=args.roll_win_sec,
            k_hi=args.roll_k_hi, k_lo=args.roll_k_lo
        )
    else:
        active = hysteresis_mask(motion_s, thr_hi, thr_lo)

    # 5) close tiny holes & enforce durations
    active_filled = fill_small_gaps(active, args.max_hole)
    segs_motion = enforce_min_on_off(active_filled, args.min_on, args.min_off)

    # 6) quality filter
    segs_motion = quality_filter(
        segs_motion, motion_s, method=args.thr,
        k_seg=args.seg_k, q_seg=args.seg_q,
        min_active_ratio=args.min_active_ratio,
        min_mean=args.min_mean
    )

    # 7) convert to frame indices
    segs = motion_segments_to_frame_segs(segs_motion, n_frames)

    # 8) apply pre/post padding in frames and merge overlaps/touches
    segs = pad_and_merge_frame_segs(segs, n_frames, args.pad_pre, args.pad_post)

    # 8b) merge close gaps by time (seconds)
    if args.merge_gap_sec and args.merge_gap_sec > 0:
        segs = merge_by_gap_seconds(segs, fps_in, max_gap_sec=args.merge_gap_sec)

    # 9) save artifacts (CSV, JSON (frames), JSON (seconds), plot)
    csv_path = os.path.join(args.out_dir, "motion.csv")
    with open(csv_path, "w") as f:
        f.write("t,motion_raw,motion_smooth,is_active\n")
        for t, (mr, ms, act) in enumerate(zip(motion_raw, motion_s, active_filled)):
            f.write(f"{t},{mr:.8f},{ms:.8f},{int(act)}\n")

    seg_json_frames_path = os.path.join(args.out_dir, "segments.json")
    with open(seg_json_frames_path, "w") as f:
        json.dump({"segments": segs, "n_frames": n_frames}, f, indent=2)

    # seconds format + full duration
    video_duration_sec = n_frames / fps_in
    timestamps = segs_to_timestamps(
        segs, fps_in, include_end=True, ndigits=3, clamp_to=video_duration_sec
    )
    seg_json_seconds_path = os.path.join(args.out_dir, "segments_sec.json")
    with open(seg_json_seconds_path, "w") as f:
        json.dump(
            {
                "duration": round(float(video_duration_sec), 3),
                "timestamps": timestamps
            },
            f,
            indent=2
        )

    # Plot
    plt.figure(figsize=(12, 4))
    x = np.arange(len(motion_s))
    plt.plot(x, motion_raw, label="motion raw", alpha=0.35)
    plt.plot(x, motion_s, label="motion smoothed", linewidth=2)
    plt.axhline(thr_hi, linestyle="--", alpha=0.7, label=f"thr_hi={thr_hi:.4g}")
    plt.axhline(thr_lo, linestyle="--", alpha=0.7, label=f"thr_lo={thr_lo:.4g}")
    # Shade active regions (after hole fill)
    in_run = False
    run_s = 0
    for i, a in enumerate(active_filled):
        if a and not in_run:
            in_run = True
            run_s = i
        if in_run and (not a or i == len(active_filled) - 1):
            run_e = i if not a else i
            plt.axvspan(run_s, run_e, alpha=0.15, color="tab:green",
                        label="active" if run_s == 0 else None)
            in_run = False

    plt.title("Motion Energy with Hysteresis Gating")
    plt.xlabel("transition index (between frames t and t+1)")
    plt.ylabel("mean flow magnitude" + (" (diag-normalized)" if args.diag_norm else " (px)"))
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "plot.png")
    plt.savefig(plot_path, dpi=150)

    print(f"\nSaved:\n- {csv_path}\n- {seg_json_frames_path}\n- {seg_json_seconds_path}\n- {plot_path}")

    # 10) optional: export clips
    if args.export_clips and segs:
        clips_dir = os.path.join(args.out_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        for i, seg in enumerate(segs):
            s, e = seg["start_frame"], seg["end_frame"]
            start_time = s / fps_in
            end_time   = (e + 1) / fps_in  # inclusive end
            out_path = os.path.join(clips_dir, f"seg_{i:04d}_{s:06d}_{e:06d}.mp4")
            export_clip_with_fallback(start_time, end_time, out_path, args.video, fps_in)


if __name__ == "__main__":
    main()
