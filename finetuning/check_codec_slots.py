# coding=utf-8
"""Check whether the codec_embedding rows used to store speaker/emotion
embeddings collide with codec tokens the talker can actually generate.

Usage:
    python3 finetuning/check_codec_slots.py --model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base
    python3 finetuning/check_codec_slots.py --model_path <path> --data_jsonl <prepared>.jsonl
"""
import argparse
import json
import os
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--start_id", type=int, default=3000)
    parser.add_argument("--num_slots", type=int, default=6)
    parser.add_argument(
        "--data_jsonl",
        type=str,
        default=None,
        help="Optional JSONL that already contains 'audio_codes' (output of prepare_data.py).",
    )
    return parser.parse_args()


def resolve_config_path(model_path):
    local = os.path.join(model_path, "config.json")
    if os.path.isfile(local):
        return local

    if os.path.isdir(model_path):
        # Some local checkouts nest the real files (e.g. HF cache snapshots).
        candidates = sorted(
            os.path.join(root, "config.json")
            for root, _dirs, files in os.walk(model_path)
            if "config.json" in files
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            print("Found several config.json files; pass the exact directory:")
            for candidate in candidates:
                print(f"  {candidate}")
            raise SystemExit(1)
        print(f"No config.json anywhere under {model_path}")
        print("Top-level contents:")
        for entry in sorted(os.listdir(model_path))[:40]:
            print(f"  {entry}")
        raise SystemExit(1)

    from huggingface_hub import snapshot_download

    resolved = snapshot_download(repo_id=model_path, local_files_only=True)
    return os.path.join(resolved, "config.json")


def report_config(config_dict, start_id, num_slots):
    talker = config_dict["talker_config"]
    vocab_size = talker["vocab_size"]
    hidden_size = talker["hidden_size"]
    eos = talker["codec_eos_token_id"]

    # Mirrors the suppress_tokens rule in Qwen3TTSForConditionalGeneration.generate
    suppress_start = vocab_size - 1024
    slots = list(range(start_id, start_id + num_slots))

    print("=" * 60)
    print("config.json")
    print("=" * 60)
    print(f"codec_embedding rows (talker vocab_size) : {vocab_size}")
    print(f"embedding dim        (talker hidden_size): {hidden_size}")
    print(f"special codec ids                        : "
          f"pad={talker['codec_pad_id']} bos={talker['codec_bos_id']} eos={eos} "
          f"think={talker['codec_think_id']} nothink={talker['codec_nothink_id']}")
    print(f"suppressed id range at generation         : [{suppress_start}, {vocab_size})")
    print(f"generatable id range                      : [0, {suppress_start}) plus eos={eos}")
    print()
    print("=" * 60)
    print(f"slots to overwrite: {slots}")
    print("=" * 60)

    unsafe = [i for i in slots if i < suppress_start]
    if not unsafe:
        print("SAFE - none of these ids can be generated; overwriting them is harmless.")
    else:
        print(f"COLLISION - these ids ARE generatable: {unsafe}")
        print("Overwriting them means a generated token of that id feeds the")
        print("speaker/emotion vector back in as its own input embedding.")
    print(f"free rows past the generatable range: [{suppress_start}, {vocab_size})")
    return suppress_start, slots


def report_usage(data_jsonl, slots):
    counter = Counter()
    total = 0
    lines = 0
    with open(data_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            codes = item.get("audio_codes")
            if codes is None:
                raise ValueError(
                    f"{data_jsonl} has no 'audio_codes'; run prepare_data.py first."
                )
            lines += 1
            for row in codes:
                total += 1
                if row[0] in slots:
                    counter[row[0]] += 1

    print()
    print("=" * 60)
    print(f"actual usage in {data_jsonl}")
    print("=" * 60)
    print(f"utterances: {lines}   codec_0 frames: {total}")
    hits = sum(counter.values())
    for slot in slots:
        print(f"  id {slot}: {counter.get(slot, 0)} frames")
    ratio = (hits / total * 100) if total else 0.0
    print(f"total hits: {hits} / {total} frames ({ratio:.4f}%)")


def main():
    args = parse_args()
    config_path = resolve_config_path(args.model_path)
    print(f"reading {config_path}\n")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    _suppress_start, slots = report_config(config_dict, args.start_id, args.num_slots)

    if args.data_jsonl:
        report_usage(args.data_jsonl, slots)


if __name__ == "__main__":
    main()
