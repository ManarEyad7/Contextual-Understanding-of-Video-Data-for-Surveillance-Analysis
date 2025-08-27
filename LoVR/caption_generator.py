#!/usr/bin/env python3
"""
new_caption_generator.py

Caption generator for surveillance clips (Qwen2-VL + vLLM).
- Supports inline few-shot examples via repeated --fewshot flags.
- Optionally also loads few-shot examples from a gt.json file.
- Deduplicates previously processed clips in --result-file.
"""

import os
import json
import random
import torch
import ffmpeg
import argparse, math, cv2, logging, pathlib
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

debug = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# -------------------- vLLM init --------------------
def llm_init(model_path: str):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,  # for 7B model
        trust_remote_code=True,
    )
    return llm
# ---------------------------------------------------


# -------------------- Helpers ----------------------
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks."""
    print(len(lst))
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def encode(x):
    """Stable id for a clip row."""
    return f'{x["vid"]}-{x["slice_num"]}'


def resize_video_to_240p(video_path, file):
    """(Optional) Example of resizing; not used in main flow."""
    global debug
    result_path = f''  # your result file path
    if debug:
        ffmpeg.input(video_path).output(
            result_path, vf="scale=426x240"
        ).overwrite_output().run()
    else:
        ffmpeg.input(video_path).output(
            result_path, vf="scale=426x240"
        ).overwrite_output().global_args('-loglevel', 'quiet').run()
    return result_path


def safe_makedirs_for_file(file_path: str):
    """Create parent folder for a file path. Guard against passing a file as a directory."""
    parent = os.path.dirname(os.path.abspath(file_path)) or "."
    # If the parent exists and is a file (e.g., someone did .../index.jsonl/captions.jsonl), error clearly.
    if os.path.exists(parent) and not os.path.isdir(parent):
        raise ValueError(
            f"[Path Error] Parent path is a file, not a folder: {parent}\n"
            f"Please set --result-file to a file inside a valid directory."
        )
    os.makedirs(parent, exist_ok=True)
# ---------------------------------------------------


# -------------------- Ground Truth (file) --------------------
def load_ground_truth(gt_file):
    """
    Load a gt.json in the format:
    {
      "VideoKey": {
        "duration": float,
        "timestamps": [[s,e], ...],
        "sentences": ["...", "...", ...]
      },
      ...
    }
    Returns a flat list of dicts: [{"time":[s,e], "caption":str}, ...]
    """
    if not gt_file:
        return []

    if not os.path.exists(gt_file):
        print(f"[GT] Ground truth file not found: {gt_file}")
        return []

    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for _vid, entry in data.items():
        tss = entry.get("timestamps", [])
        sents = entry.get("sentences", [])
        n = min(len(tss), len(sents))
        for i in range(n):
            ts = tss[i]
            sent = sents[i].strip()
            if isinstance(ts, (list, tuple)) and len(ts) == 2 and sent:
                examples.append({"time": [float(ts[0]), float(ts[1])], "caption": sent})
    print(f"[GT] Loaded {len(examples)} examples from {gt_file}")
    return examples
# -------------------------------------------------------------


# -------------------- Inline Few-shot ------------------------
def parse_inline_fewshot(fewshots):
    """
    Parse repeated --fewshot flags.
    Accepted formats:
      1) "start,end|caption text"
      2) "caption text"  (time omitted)
    Returns: [{"time":[s,e], "caption":str}, ...]
    """
    out = []
    for s in fewshots or []:
        s = s.strip()
        if '|' in s:
            ts, cap = s.split('|', 1)
            try:
                a, b = [float(x) for x in ts.split(',')]
            except Exception:
                a, b = 0.0, 0.0
            out.append({"time": [a, b], "caption": cap.strip()})
        else:
            out.append({"time": [0.0, 0.0], "caption": s})
    return out
# -------------------------------------------------------------


# -------------------- Prompt Assembly ------------------------
def build_examples_text(gt_examples, k):
    """
    Assemble k examples (random sample) into a short text block
    to prepend to the user prompt. Times are optional.
    """
    if not gt_examples or k <= 0:
        return ""

    k = min(k, len(gt_examples))
    chosen = random.sample(gt_examples, k)
    lines = ["Here are examples of how to caption surveillance clips:"]
    for ex in chosen:
        s, e = ex.get("time", [0.0, 0.0])
        cap = ex["caption"].strip()
        if (s == 0.0 and e == 0.0) or (s is None and e is None):
            lines.append(f"- {cap}")
        else:
            lines.append(f"Time {float(s):.1f}–{float(e):.1f}s → {cap}")
    lines.append("")  # blank line
    lines.append("Now caption the next clip in the same style:")
    return "\n".join(lines)
# -------------------------------------------------------------


def main(args):
    if not args.debug:
        global debug
        debug = False
        logging.getLogger("vllm").setLevel(logging.WARNING)

    # -------------------- Load few-shot examples --------------------
    gt_examples = []
    # 1) inline few-shot (highest priority)
    if args.fewshot:
        gt_examples.extend(parse_inline_fewshot(args.fewshot))
    # 2) optional gt.json
    gt_examples.extend(load_ground_truth(args.gt_file))
    # ---------------------------------------------------------------

    # -------------------- Read index.jsonl -------------------------
    with open(args.jsonl_file, 'r', encoding="utf-8") as f:
        infos = [json.loads(line) for line in f.readlines()]
    infos = get_chunk(infos, args.num_chunks, args.chunk_idx)
    # ---------------------------------------------------------------

    result_file = args.result_file
    safe_makedirs_for_file(result_file)

    if args.rerun:
        # Truncate result file (start fresh)
        with open(result_file, 'w', encoding="utf-8") as f:
            pass

    # Deduplicate / keep only relevant rows already processed
    if os.path.exists(result_file):
        processed = []
        with open(result_file, 'r', encoding="utf-8") as f:
            processed = [json.loads(line) for line in f.readlines()] if os.path.getsize(result_file) > 0 else []
        req = set([encode(i) for i in infos])
        with open(result_file, 'w', encoding="utf-8") as f:
            st = set()
            for i in processed:
                if encode(i) in st or encode(i) not in req:
                    continue
                st.add(encode(i))
                f.write(json.dumps(i) + '\n')

    with open(result_file, 'r', encoding="utf-8") as f:
        processed = [json.loads(line) for line in f.readlines()] if os.path.getsize(result_file) > 0 else []
    processed = set([encode(data) for data in processed])
    infos = [i for i in infos if encode(i) not in processed]

    # -------------------- Model & processor ------------------------
    llm = llm_init(args.model_path)
    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=512,  # keep outputs concise
        stop_token_ids=[],
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    f = open(result_file, 'a', encoding="utf-8")
    process_list = []
    print(f'{len(infos)} items to process')

    for i, info in enumerate(tqdm(infos)):
        vpath = os.path.join(args.video_folder, info['path'])
        process_list.append([i, vpath])

        if len(process_list) < args.batch_size and info != infos[-1]:
            continue

        llm_inputs = []
        valid_indices = []  # indices that made it into llm_inputs

        for j, data in enumerate(process_list):
            idx, vpath = data[0], data[1]
            abs_path = pathlib.Path(vpath).resolve().as_posix()

            if not os.path.exists(abs_path):
                tqdm.write(f'[Skip] File not found: {abs_path}')
                continue

            cap = cv2.VideoCapture(abs_path)
            if not cap.isOpened():
                tqdm.write(f'[Skip] Cannot open: {abs_path}')
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 1:
                tqdm.write(f'[Skip] Too few frames: {abs_path}')
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if frame_count < 5:
                tqdm.write(f'[Skip] <5 frames: {abs_path}')
                continue

            duration = int(frame_count / fps) if fps > 0 else 0
            fnum = min(64, max(8, duration))
            if frame_count < 8:
                fnum = 2

            # ---- Few-shot examples text (inline and/or gt.json) ----
            example_text = build_examples_text(gt_examples, args.examples_per_prompt)
            # --------------------------------------------------------

            # -------- PROMPT: Surveillance Action-Only Caption --------
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a surveillance video captioner. "
                        "Describe only observable actions relevant to security monitoring. "
                        "Focus on: who/what acted, the action verb, the target/object, and essential location cues. "
                        "Ignore scenery, colors, lighting, weather, time of day, and aesthetics unless they change or enable the action. "
                        "Be concise, factual, and avoid speculation. "
                        "If uncertain about identities, use 'person', 'group', or 'vehicle'. "
                        "Use present tense. "
                        "If no major events occur, describe any observable movement, interaction, or object motion, "
                        "instead of saying 'No action.'"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{abs_path}",
                            "nframes": fnum,
                        },
                        {
                            "type": "text",
                            "text": (
                                (example_text + "\n\n") if example_text else ""
                            ) + (
                                "Produce an action-only caption for this surveillance clip.\n\n"
                                "Rules:\n"
                                "1) Describe primary actions (e.g., 'Person enters doorway', 'Two people exchange object', 'Car stops at gate').\n"
                                "2) Include subjects, verbs, and objects (S-V-O). Mention counts if relevant (e.g., 'two people').\n"
                                "3) Mention simple location cues only if they clarify the action (e.g., 'at entrance', 'near counter').\n"
                                "4) Omit scene aesthetics, colors, weather, and camera qualities.\n"
                                "5) Do not guess identities or intentions.\n"
                                "6) If no major events occur, describe minor observable movements, interactions, or object motions.\n\n"
                                "Format:\n"
                                "Action-only caption: <one sentence (<=25 words) summarizing main actions>\n"
                                "Events:\n"
                                "- <subject – verb – object/target>\n"
                                "- <subject – verb – object/target>\n"
                                "(List up to 5 key events.)"
                            ),
                        },
                    ],
                },
            ]
            # -----------------------------------------------------------

            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            })
            valid_indices.append(idx)

        if not llm_inputs:
            process_list = []
            continue

        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

        # Map outputs back to the corresponding infos using valid_indices
        for j, idx in enumerate(valid_indices):
            generated_text = ""
            try:
                generated_text = outputs[j].outputs[0].text
            except (IndexError, AttributeError):
                print("[Warn] Empty LLM output")
                generated_text = ""
            infos[idx]['cap'] = generated_text

        # Write results for those we actually processed
        for idx in valid_indices:
            rinfo = infos[idx]
            f.write(json.dumps(rinfo, ensure_ascii=False) + '\n')
            f.flush()

        process_list = []

    f.close()
    assert len(process_list) == 0


if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"Current CUDA Device Index: {torch.cuda.current_device()}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"Device Name: {torch.cuda.get_device_name()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default='Qwen/Qwen2-VL-7B-Instruct')
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--jsonl-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # Ground-truth options (file-based)
    parser.add_argument("--gt-file", default="",
                        help="Path to gt.json; if set, a few examples are injected per prompt.")

    # Few-shot count (applies to both inline and gt.json pools)
    parser.add_argument("--examples-per-prompt", type=int, default=3,
                        help="How many examples to include in each prompt (sampled at random).")

    # Inline few-shot examples (repeatable)
    parser.add_argument(
        "--fewshot", action="append", default=[],
        help='Inline few-shot example. Use repeatedly. '
             'Formats: "start,end|caption" or "caption". '
             'Example: --fewshot "0,8|Person enters doorway." --fewshot "Two people exchange an object near counter."'
    )

    args = parser.parse_args()
    main(args)
