# coding=utf-8
"""Re-bake an exported checkpoint's emotion rows at a different strength.

`sft_12hz_Lora.py` writes `speaker_vector + emotion_scale * emotion_offset[i]`
into the codec_embedding rows registered as `<speaker>_<emotion>`, and keeps the
unscaled pieces in `emotion_table.pt`. Everything needed to rebuild those rows
therefore survives training, so trying another strength is a file rewrite rather
than another run.

Scaling only the offset leaves the speaker vector untouched, so timbre does not
move; the emotion offset is pushed further along the direction it already
encodes. This is only meaningful if the model was trained with
`--emotion_dropout`: without it the model never saw that vector vary, so it has
no behaviour to extrapolate and larger values tend to degrade rather than
intensify. Sweep upwards and listen.

Usage:
    python3 finetuning/rebake_emotion_scale.py output/final_model --scale 2.0
    python3 finetuning/rebake_emotion_scale.py output/final_model --scale 2.0 --out output/scale2
"""
import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Exported checkpoint, e.g. output/final_model")
    parser.add_argument("--scale", type=float, required=True, help="Emotion offset multiplier")
    parser.add_argument(
        "--table",
        default=None,
        help="emotion_table.pt; defaults to the parent of model_dir, where training writes it",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory; defaults to <model_dir>_scale<scale>",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    table_path = args.table or os.path.join(
        os.path.dirname(os.path.abspath(args.model_dir)), "emotion_table.pt"
    )
    if not os.path.isfile(table_path):
        raise SystemExit(f"No emotion_table.pt at {table_path}; pass --table explicitly.")

    out_dir = args.out or f"{os.path.normpath(args.model_dir)}_scale{args.scale:g}"
    if os.path.exists(out_dir):
        raise SystemExit(f"{out_dir} already exists; remove it or pass a different --out.")

    table = torch.load(table_path, map_location="cpu")
    emotions = table["emotions"]
    emotion_weight = table["emotion_weight"].float()
    speaker_vector = table["speaker_vector"].float()
    trained_scale = table.get("emotion_scale", 1.0)

    with open(os.path.join(args.model_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    # Authoritative name -> row mapping, so the slot numbers are never guessed.
    spk_id = config["talker_config"]["spk_id"]
    rows = {}
    for index, emotion in enumerate(emotions):
        matches = [row for name, row in spk_id.items() if name.lower().endswith(f"_{emotion}")]
        if len(matches) != 1:
            raise SystemExit(
                f"Expected exactly one speaker entry ending in _{emotion}, found {matches}. "
                f"spk_id = {spk_id}"
            )
        rows[index] = matches[0]

    weights_path = os.path.join(args.model_dir, "model.safetensors")
    if not os.path.isfile(weights_path):
        raise SystemExit(
            f"No single-file model.safetensors in {args.model_dir}. "
            "This script only handles the layout sft_12hz_Lora.py writes."
        )

    print(f"Re-baking {args.model_dir} at {args.scale}x (was baked at {trained_scale:g}x)")
    state_dict = load_file(weights_path)
    key = "talker.model.codec_embedding.weight"
    weight = state_dict[key]
    for index, emotion in enumerate(emotions):
        row = rows[index]
        combined = speaker_vector + args.scale * emotion_weight[index]
        weight[row] = combined.to(weight.dtype)
        offset_norm = (args.scale * emotion_weight[index]).norm().item()
        print(
            f"  row {row:>5}  {emotion:<9} offset norm {offset_norm:7.4f} "
            f"(speaker {speaker_vector.norm().item():.4f})"
        )
    state_dict[key] = weight

    shutil.copytree(args.model_dir, out_dir)
    save_file(state_dict, os.path.join(out_dir, "model.safetensors"))
    print(f"\nWrote {out_dir}")
    print("Listen before trusting the number; past a point the offset stops intensifying")
    print("the emotion and starts degrading the voice.")


if __name__ == "__main__":
    main()
