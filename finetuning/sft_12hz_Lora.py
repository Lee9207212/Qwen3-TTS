# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import csv
import itertools
import json
import random
import os
import shutil
from collections import Counter

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from dataset import EMOTION_TO_ID, EMOTIONS, TTSDataset
from huggingface_hub import snapshot_download
try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError as exc:
    raise ImportError(
        "LoRA fine-tuning requires PEFT. Install it with: pip install peft"
    ) from exc
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None

# First codec_embedding row used to store a "speaker + emotion" vector. Rows
# [SPEAKER_SLOT_START, SPEAKER_SLOT_START + len(EMOTIONS)) are overwritten at
# save time, so they must not alias anything the talker can emit or look up.
# Verified for Qwen3-TTS-12Hz-0.6B-Base: the table has 3072 rows, generation is
# restricted to ids below 2048, the special codec ids span 2148-2157 and the
# highest codec_language_id is 2071. Nothing reaches 3000. Re-check with
# check_codec_slots.py before using a different checkpoint.
SPEAKER_SLOT_START = 3000

# save_final_model() bakes one speaker vector into every exported row, taken
# from the first sample of the first batch. That is only meaningful while
# ref_audio resolves to a single fixed file for every row. Nothing downstream
# would notice if that stopped holding, so measure the spread instead of
# assuming it. Relative, so the figure is comparable across checkpoints; its
# floor is bf16 rounding, not a real difference in voice.
SPEAKER_DRIFT_TOLERANCE = 0.05
speaker_drift_in_batch = 0.0
speaker_drift_across_batches = 0.0
_speaker_drift_warned = False


def track_speaker_drift(speaker_embedding, reference):
    """Record how far apart the per-sample speaker vectors sit."""
    global speaker_drift_in_batch, speaker_drift_across_batches, _speaker_drift_warned

    vectors = speaker_embedding.detach().float()
    scale = vectors.abs().max().clamp(min=1e-6)
    in_batch = ((vectors - vectors[0:1]).abs().max() / scale).item()
    speaker_drift_in_batch = max(speaker_drift_in_batch, in_batch)

    across = 0.0
    if reference is not None:
        first = reference.detach().float().to(vectors.device)[0:1]
        across = ((vectors - first).abs().max() / scale).item()
        speaker_drift_across_batches = max(speaker_drift_across_batches, across)

    if not _speaker_drift_warned and max(in_batch, across) > SPEAKER_DRIFT_TOLERANCE:
        _speaker_drift_warned = True
        print(
            f"WARNING: speaker vectors are not constant (within batch {in_batch:.4f}, "
            f"against the first batch {across:.4f}, tolerance "
            f"{SPEAKER_DRIFT_TOLERANCE}). Every exported row reuses the first "
            "sample's vector, so the saved voice would be an arbitrary pick. "
            "Point every row at one fixed ref_audio."
        )


def report_speaker_drift(accelerator):
    """Say whether baking a single speaker vector was a safe choice."""
    accelerator.print(
        f"Speaker vector spread (relative): within a batch {speaker_drift_in_batch:.5f}, "
        f"against the first batch {speaker_drift_across_batches:.5f}, "
        f"tolerance {SPEAKER_DRIFT_TOLERANCE}"
    )
    if max(speaker_drift_in_batch, speaker_drift_across_batches) > SPEAKER_DRIFT_TOLERANCE:
        accelerator.print(
            "  WARNING: ref_audio did not produce one constant speaker vector; "
            "the baked voice is an arbitrary pick among those seen."
        )
    else:
        accelerator.print("  OK: one constant speaker vector, safe to bake.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--train_jsonl", type=str, default=None)
    parser.add_argument("--val_jsonl", type=str, default=None)
    parser.add_argument("--test_jsonl", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--emotion_lr",
        type=float,
        default=1e-3,
        help=(
            "Learning rate for the emotion table. It is a zero-initialised "
            f"{len(EMOTIONS)}x hidden_size tensor and needs a much larger step "
            "size than the LoRA adapters to move at all."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument(
        "--emotion_eval_every",
        type=int,
        default=1,
        help=(
            "Report the per-emotion validation breakdown every N epochs; 0 turns "
            "it off. The breakdown covers the same rows as the aggregate pass, so "
            "it costs one extra sweep of the validation set."
        ),
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--freeze_lora_epochs",
        type=int,
        default=0,
        help=(
            "Train the emotion table alone for this many epochs before the LoRA "
            "adapters start updating. Both share one loss, so whatever the "
            "adapters explain first is gradient the table never sees; holding "
            "them back gives the table first claim on the error signal."
        ),
    )
    parser.add_argument(
        "--emotion_dropout",
        type=float,
        default=0.0,
        help=(
            "Probability of zeroing a sample's emotion vector during training. "
            "With it the model has to behave differently with and without the "
            "offset, so it cannot make the emotion table redundant -- which is "
            "what the adapters otherwise do once they are unfrozen. It also "
            "makes --emotion_scale meaningful at inference."
        ),
    )
    parser.add_argument(
        "--emotion_scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to the emotion offsets when they are baked into "
            "the exported checkpoint. 1.0 reproduces training; above 1.0 pushes "
            "further along the direction the offset already encodes, which is a "
            "strength knob that costs no retraining. Train with --emotion_dropout "
            "first, otherwise the model has never seen the offset vary and "
            "extrapolating it is not meaningful."
        ),
    )
    parser.add_argument(
        "--select_by",
        choices=("val_loss", "emotion_delta", "emotion_shuffle"),
        default="val_loss",
        help=(
            "Checkpoint selection criterion. val_loss cannot see whether emotion "
            "conditioning works: when the transcript already implies the emotion, "
            "a dead emotion table and a working one score nearly the same. "
            "emotion_delta selects the epoch where zeroing the emotion vectors "
            "hurts validation the most, which is the effect we actually want."
        ),
    )
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated Linear module suffixes to adapt with LoRA.",
    )
    args = parser.parse_args()

    if args.dataset_dir:
        args.train_jsonl = os.path.join(args.dataset_dir, "train.jsonl")
        args.val_jsonl = os.path.join(args.dataset_dir, "val.jsonl")
        args.test_jsonl = os.path.join(args.dataset_dir, "test.jsonl")

    missing = [
        name for name in ("train_jsonl", "val_jsonl", "test_jsonl")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "Please pass --dataset_dir or all of --train_jsonl, --val_jsonl, and --test_jsonl. "
            f"Missing: {', '.join(missing)}"
        )
    return args


def apply_lora(model, args, accelerator):
    """Freeze the base model and add trainable LoRA weights to selected Linear layers."""
    target_modules = [
        name.strip()
        for name in args.lora_target_modules.split(",")
        if name.strip()
    ]
    if not target_modules:
        raise ValueError("--lora_target_modules must contain at least one module name")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        raise ValueError("--lora_rank and --lora_alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora_dropout must be in [0, 1)")

    available_linear_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    matched_linear_names = [
        name
        for name in available_linear_names
        if any(name == target or name.endswith(f".{target}") for target in target_modules)
    ]
    if not matched_linear_names:
        preview = ", ".join(available_linear_names[:30])
        raise ValueError(
            f"LoRA targets {target_modules} matched no Linear layers. "
            f"First available Linear layers: {preview}"
        )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable_parameters == 0:
        raise ValueError("LoRA was applied but no trainable parameters were found")

    accelerator.print(
        f"LoRA targets: {target_modules} | matched Linear layers: {len(matched_linear_names)}"
    )
    for name in matched_linear_names:
        accelerator.print(f"  LoRA matched: {name}")
    accelerator.print(
        f"Trainable parameters: {trainable_parameters:,}/{total_parameters:,} "
        f"({100.0 * trainable_parameters / total_parameters:.4f}%)"
    )
    return model


def set_lora_requires_grad(parameters, trainable):
    """Flip the adapters on or off. Returns True when the call changed something.

    Freezing here leaves the graph intact: the emotion table still requires grad,
    so backward walks through the frozen weights to reach it. The adapters simply
    stop accumulating .grad, and AdamW skips them.
    """
    changed = False
    for parameter in parameters:
        if parameter.requires_grad != trainable:
            parameter.requires_grad = trainable
            changed = True
    return changed


def build_emotion_table(hidden_size, accelerator):
    """A small zero-initialised lookup table, one row per emotion.

    It is deliberately kept outside the PEFT-wrapped model: anything attached
    after get_peft_model() is either frozen by PEFT or dropped by
    merge_and_unload(). Keeping it separate means we own its gradients, its
    optimizer group and its checkpointing.

    Zero initialisation makes the first training step numerically identical to
    speaker-only fine-tuning, which is a free sanity check.
    """
    table = torch.nn.Embedding(len(EMOTIONS), hidden_size)
    torch.nn.init.zeros_(table.weight)
    # Kept in fp32: bf16 cannot accumulate the small updates this table needs.
    table = table.float()
    accelerator.print(
        f"Emotion table: {len(EMOTIONS)} x {hidden_size} "
        f"({table.weight.numel():,} parameters), rows = {EMOTIONS}"
    )
    return table


def report_emotion_counts(splits, accelerator):
    """Print the per-split emotion histogram and fail fast on empty rows."""
    accelerator.print("Emotion distribution:")
    header = f"  {'split':<8}" + "".join(f"{name[:8]:>10}" for name in EMOTIONS) + f"{'total':>8}"
    accelerator.print(header)
    for split_name, rows in splits.items():
        counts = Counter(row["emotion"] for row in rows)
        line = "".join(f"{counts.get(name, 0):>10}" for name in EMOTIONS)
        accelerator.print(f"  {split_name:<8}{line}{len(rows):>8}")

    missing = [name for name in EMOTIONS if not any(
        row["emotion"] == name for row in splits["train"]
    )]
    if missing:
        raise ValueError(
            f"No training samples for {missing}; those emotion rows would never "
            "receive a gradient and would stay at their zero initialisation"
        )
    thin = [name for name in EMOTIONS if not any(
        row["emotion"] == name for row in splits["val"]
    )]
    if thin:
        accelerator.print(
            f"Validation split has no samples for {thin}; the validation loss "
            "will not reflect those emotions. Consider a stratified split."
        )


def report_emotion_table(weight, accelerator, speaker_embedding=None):
    """Print pairwise cosine similarity so we can tell whether anything was learned."""
    weight = weight.detach().float()
    norms = weight.norm(dim=-1)
    accelerator.print("Emotion vector norms:")
    for name, norm in zip(EMOTIONS, norms.tolist()):
        accelerator.print(f"  {name:<9} {norm:.6f}")
    if speaker_embedding is not None:
        speaker_norm = speaker_embedding.detach().float().norm().item()
        accelerator.print(f"  speaker   {speaker_norm:.6f} (reference scale)")

    if norms.min().item() < 1e-8:
        accelerator.print(
            "At least one emotion vector is still all zeros - it never received "
            "a gradient. Check that every emotion appears in the training split."
        )
        return

    similarity = torch.nn.functional.normalize(weight, dim=-1)
    similarity = similarity @ similarity.T
    accelerator.print("Pairwise cosine similarity (want these well below 1.0):")
    header = " " * 11 + "".join(f"{name[:7]:>9}" for name in EMOTIONS)
    accelerator.print(header)
    for i, name in enumerate(EMOTIONS):
        row = "".join(f"{similarity[i, j].item():>9.3f}" for j in range(len(EMOTIONS)))
        accelerator.print(f"  {name:<9}{row}")


def _ensure_path_exists(value, field_name, path, line_no):
    values = value if isinstance(value, list) else [value]
    for item in values:
        if not isinstance(item, str) or not os.path.exists(item):
            raise ValueError(
                f"{path}:{line_no} field '{field_name}' points to a missing path: {item}"
            )


def load_jsonl(path, split_name):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{split_name} JSONL does not exist: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"{split_name} JSONL is empty: {path}")

    required_fields = {"audio", "text", "ref_audio", "audio_codes", "emotion"}
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no} is an empty line")
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc

            missing = required_fields - item.keys()
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields: {sorted(missing)}")
            if not isinstance(item["text"], str):
                raise ValueError(f"{path}:{line_no} field 'text' must be a string")
            if item["emotion"] not in EMOTION_TO_ID:
                raise ValueError(
                    f"{path}:{line_no} field 'emotion' is {item['emotion']!r}; "
                    f"expected one of {EMOTIONS}"
                )
            _ensure_path_exists(item["audio"], "audio", path, line_no)
            _ensure_path_exists(item["ref_audio"], "ref_audio", path, line_no)

            audio_codes = item["audio_codes"]
            if (
                not isinstance(audio_codes, list)
                or not audio_codes
                or not all(isinstance(row, list) and len(row) == 16 for row in audio_codes)
            ):
                raise ValueError(
                    f"{path}:{line_no} field 'audio_codes' must be a non-empty list of 16-code rows"
                )
            rows.append(item)

    return rows


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(data, processor, config, batch_size, shuffle):
    dataset = TTSDataset(data, processor, config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)


def compute_loss(model, batch, emotion_table, zero_emotion=False,
                 shift_emotion=0, emotion_dropout=0.0):
    input_ids = batch['input_ids']
    codec_ids = batch['codec_ids']
    ref_mels = batch['ref_mels']
    emotion_ids = batch['emotion_ids']
    text_embedding_mask = batch['text_embedding_mask']
    codec_embedding_mask = batch['codec_embedding_mask']
    attention_mask = batch['attention_mask']
    codec_0_labels = batch['codec_0_labels']
    codec_mask = batch['codec_mask']

    # Detached on purpose: the speaker encoder stays frozen, so with a fixed
    # ref_audio this vector is a constant for the whole run.
    speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
    track_speaker_drift(speaker_embedding, target_speaker_embedding)
    emotion_ids = emotion_ids.to(model.device)
    if shift_emotion:
        # Hand every sample a different emotion's vector. This is a cyclic
        # permutation, so each vector is still used exactly as often as before
        # and only the sample-to-vector pairing changes; a loss increase cannot
        # be blamed on some vector being over-represented. The shift is never a
        # multiple of len(EMOTIONS), so no sample keeps its own vector.
        shifted = (emotion_ids + shift_emotion) % len(EMOTIONS)
        emotion_ids = torch.where(emotion_ids >= 0, shifted, emotion_ids)
    # NOT detached: this is the only path by which the emotion table learns.
    emotion_embedding = emotion_table(emotion_ids)
    if emotion_dropout > 0.0:
        # Per sample, not per batch. The model sees both conditions during
        # training and has to tell them apart.
        keep = (
            torch.rand(emotion_embedding.shape[0], device=emotion_embedding.device)
            >= emotion_dropout
        )
        emotion_embedding = emotion_embedding * keep.unsqueeze(-1)
    # Ablation for the emotion_delta metric: keep every other input identical and
    # drop only the emotion offset, so the loss difference is attributable to it.
    if zero_emotion:
        emotion_embedding = torch.zeros_like(emotion_embedding)

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    input_text_embedding = model.talker.text_projection(
        model.talker.model.text_embedding(input_text_ids)
    ) * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    # Slot 6 is masked to zero by the collate function and is the model's only
    # conditioning vector; speaker identity and emotion share it additively.
    input_codec_embedding[:, 6, :] = speaker_embedding + emotion_embedding.to(speaker_embedding.dtype)

    input_embeddings = input_text_embedding + input_codec_embedding

    for i in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    outputs = model.talker(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
        labels=codec_0_labels,
        output_hidden_states=True
    )

    hidden_states = outputs.hidden_states[0][-1][:, :-1, :]
    talker_hidden_states = hidden_states[codec_mask[:, 1:]]
    talker_codec_ids = codec_ids[codec_mask]

    sub_talker_logits, _sub_talker_loss = model.talker.forward_sub_talker_finetune(
        talker_codec_ids, talker_hidden_states
    )
    sub_talker_targets = talker_codec_ids[:, 1:]
    if sub_talker_logits.shape[:-1] != sub_talker_targets.shape:
        raise ValueError(
            "Sub-talker logits/targets are misaligned: "
            f"logits={tuple(sub_talker_logits.shape)}, "
            f"targets={tuple(sub_talker_targets.shape)}"
        )
    sub_talker_loss = F.cross_entropy(
        sub_talker_logits.reshape(-1, sub_talker_logits.shape[-1]).float(),
        sub_talker_targets.reshape(-1).to(sub_talker_logits.device),
    )

    loss = outputs.loss + 0.3 * sub_talker_loss

    main_shift_labels = codec_0_labels[:, 1:]
    main_valid_tokens = (main_shift_labels != -100).sum()
    sub_valid_tokens = sub_talker_targets.numel()
    stats = {
        "main_loss_sum": outputs.loss.detach().float() * main_valid_tokens.float(),
        "main_tokens": main_valid_tokens.detach().float(),
        "sub_loss_sum": sub_talker_loss.detach().float() * torch.tensor(
            sub_valid_tokens, device=sub_talker_loss.device, dtype=torch.float32
        ),
        "sub_tokens": torch.tensor(sub_valid_tokens, device=sub_talker_loss.device, dtype=torch.float32),
    }
    return loss, stats, speaker_embedding


def _empty_loss_stats():
    return {
        "main_loss_sum": 0.0,
        "main_tokens": 0.0,
        "sub_loss_sum": 0.0,
        "sub_tokens": 0.0,
    }


def _gather_scalar_sum(accelerator, value):
    tensor = value.detach().reshape(1).to(accelerator.device, dtype=torch.float32)
    return accelerator.gather(tensor).sum().item()


def add_loss_stats(total_stats, batch_stats, accelerator):
    for key, value in batch_stats.items():
        total_stats[key] += _gather_scalar_sum(accelerator, value)


def finalize_loss(total_stats):
    if total_stats["main_tokens"] <= 0 or total_stats["sub_tokens"] <= 0:
        raise ValueError("Cannot compute loss because no valid label tokens were found")
    main_loss = total_stats["main_loss_sum"] / total_stats["main_tokens"]
    sub_loss = total_stats["sub_loss_sum"] / total_stats["sub_tokens"]
    total_loss = main_loss + 0.3 * sub_loss
    return {
        "main_loss": main_loss,
        "sub_talker_loss": sub_loss,
        "total_loss": total_loss,
    }


@torch.no_grad()
def evaluate(model, dataloader, accelerator, emotion_table, max_batches=None,
             zero_emotion=False, shift_emotion=0):
    model.eval()
    emotion_table.eval()
    total_stats = _empty_loss_stats()
    for step, batch in enumerate(dataloader):
        if max_batches is not None and step >= max_batches:
            break
        _loss, batch_stats, _speaker_embedding = compute_loss(
            model, batch, emotion_table,
            zero_emotion=zero_emotion, shift_emotion=shift_emotion,
        )
        add_loss_stats(total_stats, batch_stats, accelerator)
    model.train()
    emotion_table.train()
    return finalize_loss(total_stats)


def build_emotion_dataloaders(splits, processor, config, batch_size, accelerator):
    """One eval dataloader per (split, emotion), built and prepared once.

    `accelerator.prepare` keeps a reference to every dataloader handed to it, so
    building these per epoch would pile up. Emotions absent from a split are
    skipped rather than producing an empty loader.
    """
    keys = []
    loaders = []
    for split_name, rows in splits.items():
        for emotion in EMOTIONS:
            subset = [row for row in rows if row["emotion"] == emotion]
            if not subset:
                continue
            keys.append((split_name, emotion))
            loaders.append(
                build_dataloader(subset, processor, config, batch_size, shuffle=False)
            )
    if not loaders:
        return {}
    prepared = accelerator.prepare(*loaders)
    if len(loaders) == 1:
        prepared = (prepared,)
    return dict(zip(keys, prepared))


@torch.no_grad()
def evaluate_by_emotion(model, emotion_loaders, split, accelerator, emotion_table, max_batches=None):
    """Per-emotion losses for one split.

    Each emotion gets its own homogeneous pass so `evaluate` is reused
    unchanged and the numbers stay comparable with the aggregate figure.
    Splitting the batched mean per token would not be comparable, because a
    batch mixes emotions and the mean is already reduced.
    """
    results = {}
    for emotion in EMOTIONS:
        loader = emotion_loaders.get((split, emotion))
        results[emotion] = (
            evaluate(model, loader, accelerator, emotion_table, max_batches)
            if loader is not None
            else None
        )
    return results


def report_emotion_losses(results, accelerator, title):
    """Print one loss row per emotion, then name the best and the worst.

    Support is uneven - fear has the fewest training rows and surprise the most
    - so an emotion that never fitted can hide behind the others in a single
    averaged number.
    """
    accelerator.print(f"{title} by emotion:")
    accelerator.print(f"  {'emotion':<10}{'main':>11}{'sub':>11}{'total':>11}")
    ranked = []
    for emotion in EMOTIONS:
        losses = results.get(emotion)
        if losses is None:
            accelerator.print(f"  {emotion:<10}{'(no rows)':>11}")
            continue
        accelerator.print(
            f"  {emotion:<10}{losses['main_loss']:>11.4f}"
            f"{losses['sub_talker_loss']:>11.4f}{losses['total_loss']:>11.4f}"
        )
        ranked.append((losses["total_loss"], emotion))
    if len(ranked) > 1:
        ranked.sort()
        accelerator.print(
            f"  best {ranked[0][1]} {ranked[0][0]:.4f} | "
            f"worst {ranked[-1][1]} {ranked[-1][0]:.4f}"
        )


def write_emotion_loss_rows(writer, csv_file, record_type, epoch, global_step, split, results):
    """Long format, one row per (split, emotion), so adding an emotion later
    needs no schema change."""
    if writer is None:
        return
    for emotion in EMOTIONS:
        losses = results.get(emotion)
        if losses is None:
            continue
        writer.writerow(
            {
                "record_type": record_type,
                "epoch": epoch,
                "global_step": global_step,
                "split": split,
                "emotion": emotion,
                "main_loss": losses["main_loss"],
                "sub_talker_loss": losses["sub_talker_loss"],
                "total_loss": losses["total_loss"],
            }
        )
    csv_file.flush()


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def clone_model_state(model, accelerator):
    unwrapped_model = accelerator.unwrap_model(model)
    return {
        key: value.detach().cpu().clone()
        for key, value in unwrapped_model.state_dict().items()
    }


def load_model_state(model, accelerator, state_dict):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(state_dict)


def prepare_output_dir(output_model_path):
    os.makedirs(output_model_path, exist_ok=True)
    protected_paths = [
        os.path.join(output_model_path, "loss_history.csv"),
        os.path.join(output_model_path, "final_model"),
    ]
    existing = [path for path in protected_paths if os.path.exists(path)]
    if existing:
        raise FileExistsError(
            "Output directory already contains experiment artifacts; please use a new --output_model_path. "
            f"Existing: {existing}"
        )


def write_loss_row(
    writer,
    csv_file,
    record_type,
    epoch,
    global_step,
    train_losses,
    val_losses,
    test_losses,
    current_lr_value,
    gradient_norm=None,
    has_nan_or_inf=False,
    emotion_delta=None,
    emotion_shuffle_delta=None,
):
    def get_loss(losses, key):
        if losses is None:
            return None
        return losses[key]

    def fmt_loss(value):
        return "" if value is None else f"{value:.6f}"

    def fmt_lr(value):
        return "" if value is None else f"{value:.10g}"

    writer.writerow({
        "record_type": record_type,
        "epoch": epoch,
        "global_step": global_step,
        "train_main_loss": fmt_loss(get_loss(train_losses, "main_loss")),
        "train_sub_talker_loss": fmt_loss(get_loss(train_losses, "sub_talker_loss")),
        "train_total_loss": fmt_loss(get_loss(train_losses, "total_loss")),
        "val_main_loss": fmt_loss(get_loss(val_losses, "main_loss")),
        "val_sub_talker_loss": fmt_loss(get_loss(val_losses, "sub_talker_loss")),
        "val_total_loss": fmt_loss(get_loss(val_losses, "total_loss")),
        "val_emotion_delta": fmt_loss(emotion_delta),
        "val_emotion_shuffle_delta": fmt_loss(emotion_shuffle_delta),
        "test_main_loss": fmt_loss(get_loss(test_losses, "main_loss")),
        "test_sub_talker_loss": fmt_loss(get_loss(test_losses, "sub_talker_loss")),
        "test_total_loss": fmt_loss(get_loss(test_losses, "total_loss")),
        "current_lr": fmt_lr(current_lr_value),
        "gradient_norm": fmt_loss(gradient_norm),
        "has_nan_or_inf": str(bool(has_nan_or_inf)).lower(),
    })
    csv_file.flush()


def save_final_model(accelerator, model, emotion_table, model_path, output_model_path,
                     speaker_name, emotion_scale=1.0):
    if not accelerator.is_main_process:
        return
    source_model_path = model_path
    if not os.path.isdir(source_model_path):
        accelerator.print(
            f"Resolving Hugging Face model '{model_path}' from the local cache for final saving."
        )
        source_model_path = snapshot_download(
            repo_id=model_path,
            local_files_only=True,
        )
    if target_speaker_embedding is None:
        raise ValueError("No target speaker embedding was captured during training")

    output_dir = os.path.join(output_model_path, "final_model")
    shutil.copytree(source_model_path, output_dir)

    input_config_file = os.path.join(source_model_path, "config.json")
    output_config_file = os.path.join(output_dir, "config.json")
    with open(input_config_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})

    # One registered speaker name per emotion. Inference performs a single
    # embedding lookup, so the "speaker + emotion" sum is precomputed here and
    # stored as six separate rows; generate_custom_voice() needs no changes.
    emotion_speaker_names = {
        emotion: f"{speaker_name}_{emotion}" for emotion in EMOTIONS
    }
    # Registered lower-cased on purpose. The released qwen-tts package looks a
    # speaker up as `spk_id[speaker.lower()]`, so a key like "F2_anger" is
    # unreachable there and generation fails with "Speaker ... not implemented".
    # Lower-cased keys resolve under that lookup and under the case-insensitive
    # one in this repo, so the exported checkpoint works with either version.
    # Callers still pass whatever casing they like.
    talker_config["spk_id"] = {
        emotion_speaker_names[emotion].lower(): SPEAKER_SLOT_START + index
        for index, emotion in enumerate(EMOTIONS)
    }
    talker_config["spk_is_dialect"] = {
        name.lower(): False for name in emotion_speaker_names.values()
    }
    config_dict["talker_config"] = talker_config
    config_dict["emotion_speakers"] = emotion_speaker_names

    with open(output_config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    unwrapped_model = accelerator.unwrap_model(model)
    if isinstance(unwrapped_model, PeftModel):
        # Restore the original Qwen3-TTS parameter names so the existing
        # Qwen3TTSModel.from_pretrained inference path can load this checkpoint.
        unwrapped_model = unwrapped_model.merge_and_unload()
    state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

    drop_prefix = "speaker_encoder"
    keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
    for k in keys_to_drop:
        del state_dict[k]

    weight = state_dict['talker.model.codec_embedding.weight']
    last_slot = SPEAKER_SLOT_START + len(EMOTIONS)
    if last_slot > weight.shape[0]:
        raise ValueError(
            f"codec_embedding has {weight.shape[0]} rows but slots up to {last_slot} "
            "are required; lower SPEAKER_SLOT_START or reduce the emotion count"
        )

    speaker_vector = target_speaker_embedding[0].detach().to("cpu").float()
    emotion_weight = accelerator.unwrap_model(emotion_table).weight.detach().to("cpu").float()

    report_speaker_drift(accelerator)
    report_emotion_table(emotion_weight, accelerator, speaker_vector)

    if emotion_scale != 1.0:
        accelerator.print(
            f"Baking emotion offsets at {emotion_scale}x. The speaker vector is not "
            "scaled, so timbre is unchanged and only the emotion offset is pushed "
            "further along its own direction."
        )
    for index, emotion in enumerate(EMOTIONS):
        combined = speaker_vector + emotion_scale * emotion_weight[index]
        weight[SPEAKER_SLOT_START + index] = combined.to(weight.device).to(weight.dtype)
        accelerator.print(
            f"codec_embedding[{SPEAKER_SLOT_START + index}] <- "
            f"{emotion_speaker_names[emotion]}"
        )

    save_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, save_path)

    # The raw table is not part of the model; keep it for inspection and for
    # rebuilding the rows without retraining.
    torch.save(
        {
            "emotions": EMOTIONS,
            "emotion_weight": emotion_weight,
            "speaker_vector": speaker_vector,
            "emotion_scale": emotion_scale,
        },
        os.path.join(output_model_path, "emotion_table.pt"),
    )


def train():
    global target_speaker_embedding

    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    if accelerator.is_main_process:
        prepare_output_dir(args.output_model_path)
    accelerator.wait_for_everyone()

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    qwen3tts.model = apply_lora(qwen3tts.model, args, accelerator)
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = load_jsonl(args.train_jsonl, "train")
    val_data = load_jsonl(args.val_jsonl, "validation")
    test_data = load_jsonl(args.test_jsonl, "test")
    report_emotion_counts(
        {"train": train_data, "val": val_data, "test": test_data}, accelerator
    )
    train_dataloader = build_dataloader(train_data, qwen3tts.processor, config, args.batch_size, shuffle=True)
    val_dataloader = build_dataloader(val_data, qwen3tts.processor, config, args.batch_size, shuffle=False)
    test_dataloader = build_dataloader(test_data, qwen3tts.processor, config, args.batch_size, shuffle=False)

    emotion_table = build_emotion_table(config.talker_config.hidden_size, accelerator)

    trainable_parameters = [
        parameter
        for parameter in qwen3tts.model.parameters()
        if parameter.requires_grad
    ]
    optimizer = AdamW(
        [
            {"params": trainable_parameters, "lr": args.lr},
            # No weight decay: decaying towards zero would actively erase the
            # emotion offsets we are trying to learn.
            {"params": list(emotion_table.parameters()), "lr": args.emotion_lr, "weight_decay": 0.0},
        ],
        lr=args.lr,
        weight_decay=0.01,
    )

    model, emotion_table, optimizer, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
        qwen3tts.model, emotion_table, optimizer, train_dataloader, val_dataloader, test_dataloader
    )
    # Re-resolved from the prepared model rather than reusing the list built
    # above: --freeze_lora_epochs toggles requires_grad on these objects, and
    # they have to be the ones the forward pass actually uses. The emotion table
    # is prepared separately, so it is not in here and never gets frozen.
    lora_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    emotion_loaders = (
        build_emotion_dataloaders(
            {"val": val_data, "test": test_data},
            qwen3tts.processor,
            config,
            args.batch_size,
            accelerator,
        )
        if args.emotion_eval_every > 0
        else {}
    )
    num_epochs = args.num_epochs
    model.train()
    emotion_table.train()
    global_step = 0
    best_val_loss = None
    best_emotion_delta = None
    best_score = None
    best_epoch = 0
    best_model_state = None
    best_emotion_state = None
    completed_epochs = 0

    csv_file = None
    writer = None
    if accelerator.is_main_process:
        csv_path = os.path.join(args.output_model_path, "loss_history.csv")
        csv_file = open(csv_path, "w", encoding="utf-8", newline="")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "record_type",
                "epoch",
                "global_step",
                "train_main_loss",
                "train_sub_talker_loss",
                "train_total_loss",
                "val_main_loss",
                "val_sub_talker_loss",
                "val_total_loss",
                "val_emotion_delta",
                "val_emotion_shuffle_delta",
                "test_main_loss",
                "test_sub_talker_loss",
                "test_total_loss",
                "current_lr",
                "gradient_norm",
                "has_nan_or_inf",
            ],
        )
        writer.writeheader()
        csv_file.flush()

    emotion_csv_file = None
    emotion_writer = None
    if accelerator.is_main_process and emotion_loaders:
        emotion_csv_path = os.path.join(args.output_model_path, "emotion_loss_history.csv")
        emotion_csv_file = open(emotion_csv_path, "w", encoding="utf-8", newline="")
        emotion_writer = csv.DictWriter(
            emotion_csv_file,
            fieldnames=[
                "record_type",
                "epoch",
                "global_step",
                "split",
                "emotion",
                "main_loss",
                "sub_talker_loss",
                "total_loss",
            ],
        )
        emotion_writer.writeheader()
        emotion_csv_file.flush()

    for epoch in range(num_epochs):
        lora_active = epoch >= args.freeze_lora_epochs
        if set_lora_requires_grad(lora_parameters, lora_active):
            accelerator.print(
                f"Epoch {epoch + 1}: LoRA adapters "
                f"{'unfrozen, both groups now train' if lora_active else 'frozen, emotion table trains alone'}"
            )
        train_stats = _empty_loss_stats()
        epoch_gradient_norm_sum = 0.0
        epoch_gradient_norm_count = 0
        epoch_has_nan_or_inf = False
        for step, batch in enumerate(train_dataloader):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break
            with accelerator.accumulate(model):
                loss, batch_stats, speaker_embedding = compute_loss(
                    model, batch, emotion_table,
                    emotion_dropout=args.emotion_dropout,
                )
                loss_is_finite = torch.isfinite(loss.detach()).all()
                if not bool(loss_is_finite.item()):
                    epoch_has_nan_or_inf = True
                    raise ValueError(
                        f"NaN/Inf detected in training loss at epoch {epoch + 1}, step {step}"
                    )
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        itertools.chain(model.parameters(), emotion_table.parameters()), 1.0
                    )
                    grad_norm_value = float(grad_norm.detach().float().item())
                    if not torch.isfinite(grad_norm.detach()).all():
                        epoch_has_nan_or_inf = True
                    epoch_gradient_norm_sum += grad_norm_value
                    epoch_gradient_norm_count += 1

                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    global_step += 1

            add_loss_stats(train_stats, batch_stats, accelerator)

            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch + 1}/{num_epochs} | Global step {global_step} | "
                    f"Step {step} | Current LR {current_lr(optimizer):.10g} | "
                    f"Training Loss: {loss.detach().float().item():.6f}"
                )

        train_losses = finalize_loss(train_stats)
        val_losses = evaluate(model, val_dataloader, accelerator, emotion_table, args.max_eval_batches)
        # Same rows, same weights, emotion offsets zeroed. How much worse the
        # model gets is how much the emotion table is actually contributing.
        # main_loss only: the sub-talker term sits on a plateau and would just
        # add noise to the difference.
        val_losses_ablated = evaluate(
            model, val_dataloader, accelerator, emotion_table, args.max_eval_batches,
            zero_emotion=True,
        )
        emotion_delta = val_losses_ablated["main_loss"] - val_losses["main_loss"]
        # Every sample gets another emotion's vector. Zeroing asks whether the
        # table contributes at all; this asks whether the six vectors are
        # distinguishable, which is the thing emotion control actually needs.
        # Six identical vectors would score a large zero-delta and a shuffle
        # delta of zero, and only the second answer would be right.
        val_losses_shuffled = evaluate(
            model, val_dataloader, accelerator, emotion_table, args.max_eval_batches,
            shift_emotion=1,
        )
        emotion_shuffle_delta = val_losses_shuffled["main_loss"] - val_losses["main_loss"]
        train_loss = train_losses["total_loss"]
        val_loss = val_losses["total_loss"]
        epoch_gradient_norm = (
            epoch_gradient_norm_sum / epoch_gradient_norm_count
            if epoch_gradient_norm_count > 0 else None
        )
        if args.select_by == "emotion_shuffle":
            score = emotion_shuffle_delta
            improved = best_score is None or score > best_score
        elif args.select_by == "emotion_delta":
            score, improved = emotion_delta, best_score is None or emotion_delta > best_score
        else:
            score, improved = val_loss, best_score is None or val_loss < best_score
        if improved:
            best_score = score
            best_val_loss = val_loss
            best_emotion_delta = emotion_delta
            best_epoch = epoch + 1
            best_model_state = clone_model_state(model, accelerator)
            best_emotion_state = clone_model_state(emotion_table, accelerator)

        lr = current_lr(optimizer)
        completed_epochs = epoch + 1

        accelerator.print(
            f"Epoch {epoch + 1}/{num_epochs} | Global step {global_step} | "
            f"Current LR {lr:.10g} | "
            f"Train main/sub/total: {train_losses['main_loss']:.6f}/"
            f"{train_losses['sub_talker_loss']:.6f}/{train_losses['total_loss']:.6f} | "
            f"Val main/sub/total: {val_losses['main_loss']:.6f}/"
            f"{val_losses['sub_talker_loss']:.6f}/{val_losses['total_loss']:.6f} | "
            f"Grad norm: {epoch_gradient_norm if epoch_gradient_norm is not None else 'NA'} | "
            f"Emotion delta zero/shuffle: {emotion_delta:+.6f}/{emotion_shuffle_delta:+.6f} | "
            f"LoRA: {'on' if lora_active else 'frozen'} | "
            f"NaN/Inf: {epoch_has_nan_or_inf} | "
            f"Best by {args.select_by}: val {best_val_loss:.6f}, "
            f"delta {best_emotion_delta:+.6f} (epoch {best_epoch})"
        )
        if accelerator.is_main_process:
            write_loss_row(
                writer,
                csv_file,
                "epoch",
                epoch + 1,
                global_step,
                train_losses,
                val_losses,
                None,
                lr,
                epoch_gradient_norm,
                epoch_has_nan_or_inf,
                emotion_delta=emotion_delta,
                emotion_shuffle_delta=emotion_shuffle_delta,
            )

        if emotion_loaders and (epoch + 1) % args.emotion_eval_every == 0:
            val_emotion_losses = evaluate_by_emotion(
                model, emotion_loaders, "val", accelerator, emotion_table, args.max_eval_batches
            )
            report_emotion_losses(
                val_emotion_losses, accelerator, f"Epoch {epoch + 1} validation"
            )
            if accelerator.is_main_process:
                write_emotion_loss_rows(
                    emotion_writer,
                    emotion_csv_file,
                    "epoch",
                    epoch + 1,
                    global_step,
                    "val",
                    val_emotion_losses,
                )

    if best_model_state is None:
        raise ValueError("No validation result was recorded; cannot restore best model for final test")
    accelerator.print(
        f"Restoring best model from epoch {best_epoch} (selected by {args.select_by}) "
        f"with Validation Loss {best_val_loss:.6f} and emotion delta "
        f"{best_emotion_delta:+.6f} before final test."
    )
    load_model_state(model, accelerator, best_model_state)
    load_model_state(emotion_table, accelerator, best_emotion_state)
    model.train()
    emotion_table.train()

    test_losses = evaluate(model, test_dataloader, accelerator, emotion_table, args.max_eval_batches)
    test_loss = test_losses["total_loss"]
    lr = current_lr(optimizer)
    accelerator.print(
        f"Final Test | Epoch {completed_epochs} | Best epoch {best_epoch} | Global step {global_step} | "
        f"Current LR {lr:.10g} | Test main/sub/total: {test_losses['main_loss']:.6f}/"
        f"{test_losses['sub_talker_loss']:.6f}/{test_losses['total_loss']:.6f}"
    )
    test_emotion_losses = {}
    if emotion_loaders:
        test_emotion_losses = evaluate_by_emotion(
            model, emotion_loaders, "test", accelerator, emotion_table, args.max_eval_batches
        )
        report_emotion_losses(test_emotion_losses, accelerator, "Final test")

    if accelerator.is_main_process:
        write_loss_row(
            writer,
            csv_file,
            "final_test",
            completed_epochs,
            global_step,
            None,
            None,
            test_losses,
            lr,
            None,
            False,
        )
        csv_file.close()
        if emotion_csv_file is not None:
            write_emotion_loss_rows(
                emotion_writer,
                emotion_csv_file,
                "final_test",
                completed_epochs,
                global_step,
                "test",
                test_emotion_losses,
            )
            emotion_csv_file.close()

    accelerator.wait_for_everyone()
    save_final_model(
        accelerator, model, emotion_table, MODEL_PATH, args.output_model_path,
        args.speaker_name, emotion_scale=args.emotion_scale,
    )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
