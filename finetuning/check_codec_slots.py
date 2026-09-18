# coding=utf-8
"""Check whether the codec_embedding rows used to store speaker/emotion
embeddings collide with codec tokens the talker can actually generate.

Reads the codec embedding shape from config.json when available, and falls
back to the safetensors header (no weights are loaded either way).

Usage:
    python3 finetuning/check_codec_slots.py --model_path <model dir>
    python3 finetuning/check_codec_slots.py --model_path <model dir> --data_jsonl <prepared>.jsonl
"""
import argparse
import json
import os
import struct
from collections import Counter

CODEC_EMBEDDING_KEY = "talker.model.codec_embedding.weight"


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


def find_file(model_path, filename):
    """Locate `filename` at the top level of model_path, else anywhere beneath it."""
    direct = os.path.join(model_path, filename)
    if os.path.isfile(direct):
        return direct
    if not os.path.isdir(model_path):
        return None
    for root, _dirs, files in os.walk(model_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def read_safetensors_header(path):
    """Return the safetensors JSON header without reading any tensor data."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len).decode("utf-8"))


def codec_embedding_shape(model_path):
    """(rows, dim) of the codec embedding, from config.json or the weight file."""
    config_path = find_file(model_path, "config.json")
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        talker = config_dict["talker_config"]
        print(f"source: {config_path}")
        return talker["vocab_size"], talker["hidden_size"], talker

    weights_path = find_file(model_path, "model.safetensors")
    if weights_path is None:
        print(f"Neither config.json nor model.safetensors found under {model_path}")
        print("Top-level contents:")
        for entry in sorted(os.listdir(model_path))[:40]:
            print(f"  {entry}")
        raise SystemExit(1)

    header = read_safetensors_header(weights_path)
    if CODEC_EMBEDDING_KEY not in header:
        matches = [k for k in header if "codec_embedding" in k]
        print(f"'{CODEC_EMBEDDING_KEY}' not in {weights_path}")
        print("Keys containing 'codec_embedding':")
        for key in matches[:20] or ["(none)"]:
            print(f"  {key}")
        raise SystemExit(1)

    rows, dim = header[CODEC_EMBEDDING_KEY]["shape"]
    print(f"source: {weights_path} (no config.json; shape read from the weight header)")
    return rows, dim, None


def report(rows, dim, talker, start_id, num_slots):
    # Mirrors the suppress_tokens rule in Qwen3TTSForConditionalGeneration.generate
    suppress_start = rows - 1024
    slots = list(range(start_id, start_id + num_slots))

    print()
    print("=" * 62)
    print(f"codec_embedding rows : {rows}")
    print(f"embedding dim        : {dim}")
    if talker is not None:
        print(
            f"special codec ids    : pad={talker['codec_pad_id']} bos={talker['codec_bos_id']} "
            f"eos={talker['codec_eos_token_id']}"
        )
    print(f"suppressed at generation : [{suppress_start}, {rows})")
    print(f"generatable              : [0, {suppress_start})")
    print("=" * 62)
    print(f"slots to overwrite: {slots}")
    unsafe = [i for i in slots if i < suppress_start]
    if unsafe:
        print(f"COLLISION - these ids ARE generatable: {unsafe}")
        print("A generated token with that id would read the speaker/emotion")
        print("vector as its own input embedding.")
    else:
        print("SAFE - none of these ids can be generated.")
    print(f"rows past the generatable range: [{suppress_start}, {rows})")
    return slots


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
    print("=" * 62)
    print(f"actual usage in {data_jsonl}")
    print("=" * 62)
    print(f"utterances: {lines}   codec_0 frames: {total}")
    for slot in slots:
        print(f"  id {slot}: {counter.get(slot, 0)} frames")
    hits = sum(counter.values())
    ratio = (hits / total * 100) if total else 0.0
    print(f"total hits: {hits} / {total} frames ({ratio:.4f}%)")


def main():
    args = parse_args()
    rows, dim, talker = codec_embedding_shape(args.model_path)
    slots = report(rows, dim, talker, args.start_id, args.num_slots)
    if args.data_jsonl:
        report_usage(args.data_jsonl, slots)


if __name__ == "__main__":
    main()
