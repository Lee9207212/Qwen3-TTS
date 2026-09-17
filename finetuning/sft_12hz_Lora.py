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
import json
import random
import os
import shutil

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from dataset import TTSDataset
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
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=41)
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

    required_fields = {"audio", "text", "ref_audio", "audio_codes"}
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


def compute_loss(model, batch):
    input_ids = batch['input_ids']
    codec_ids = batch['codec_ids']
    ref_mels = batch['ref_mels']
    text_embedding_mask = batch['text_embedding_mask']
    codec_embedding_mask = batch['codec_embedding_mask']
    attention_mask = batch['attention_mask']
    codec_0_labels = batch['codec_0_labels']
    codec_mask = batch['codec_mask']

    speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    input_text_embedding = model.talker.text_projection(
        model.talker.model.text_embedding(input_text_ids)
    ) * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    input_codec_embedding[:, 6, :] = speaker_embedding

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
def evaluate(model, dataloader, accelerator, max_batches=None):
    model.eval()
    total_stats = _empty_loss_stats()
    for step, batch in enumerate(dataloader):
        if max_batches is not None and step >= max_batches:
            break
        _loss, batch_stats, _speaker_embedding = compute_loss(model, batch)
        add_loss_stats(total_stats, batch_stats, accelerator)
    model.train()
    return finalize_loss(total_stats)


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
        "test_main_loss": fmt_loss(get_loss(test_losses, "main_loss")),
        "test_sub_talker_loss": fmt_loss(get_loss(test_losses, "sub_talker_loss")),
        "test_total_loss": fmt_loss(get_loss(test_losses, "total_loss")),
        "current_lr": fmt_lr(current_lr_value),
        "gradient_norm": fmt_loss(gradient_norm),
        "has_nan_or_inf": str(bool(has_nan_or_inf)).lower(),
    })
    csv_file.flush()


def save_final_model(accelerator, model, model_path, output_model_path, speaker_name):
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
    talker_config["spk_id"] = {
        speaker_name: 3000
    }
    talker_config["spk_is_dialect"] = {
        speaker_name: False
    }
    config_dict["talker_config"] = talker_config

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
    state_dict['talker.model.codec_embedding.weight'][3000] = (
        target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
    )
    save_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, save_path)


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
    train_dataloader = build_dataloader(train_data, qwen3tts.processor, config, args.batch_size, shuffle=True)
    val_dataloader = build_dataloader(val_data, qwen3tts.processor, config, args.batch_size, shuffle=False)
    test_dataloader = build_dataloader(test_data, qwen3tts.processor, config, args.batch_size, shuffle=False)

    trainable_parameters = [
        parameter
        for parameter in qwen3tts.model.parameters()
        if parameter.requires_grad
    ]
    optimizer = AdamW(trainable_parameters, lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader, val_dataloader, test_dataloader
    )
    num_epochs = args.num_epochs
    model.train()
    global_step = 0
    best_val_loss = None
    best_epoch = 0
    best_model_state = None
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

    for epoch in range(num_epochs):
        train_stats = _empty_loss_stats()
        epoch_gradient_norm_sum = 0.0
        epoch_gradient_norm_count = 0
        epoch_has_nan_or_inf = False
        for step, batch in enumerate(train_dataloader):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break
            with accelerator.accumulate(model):
                loss, batch_stats, speaker_embedding = compute_loss(model, batch)
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
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
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
        val_losses = evaluate(model, val_dataloader, accelerator, args.max_eval_batches)
        train_loss = train_losses["total_loss"]
        val_loss = val_losses["total_loss"]
        epoch_gradient_norm = (
            epoch_gradient_norm_sum / epoch_gradient_norm_count
            if epoch_gradient_norm_count > 0 else None
        )
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = clone_model_state(model, accelerator)

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
            f"NaN/Inf: {epoch_has_nan_or_inf} | Best Val Loss: {best_val_loss:.6f} "
            f"(epoch {best_epoch})"
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
            )

    if best_model_state is None:
        raise ValueError("No validation result was recorded; cannot restore best model for final test")
    accelerator.print(
        f"Restoring best model from epoch {best_epoch} with Validation Loss {best_val_loss:.6f} "
        "before final test."
    )
    load_model_state(model, accelerator, best_model_state)
    model.train()

    test_losses = evaluate(model, test_dataloader, accelerator, args.max_eval_batches)
    test_loss = test_losses["total_loss"]
    lr = current_lr(optimizer)
    accelerator.print(
        f"Final Test | Epoch {completed_epochs} | Best epoch {best_epoch} | Global step {global_step} | "
        f"Current LR {lr:.10g} | Test main/sub/total: {test_losses['main_loss']:.6f}/"
        f"{test_losses['sub_talker_loss']:.6f}/{test_losses['total_loss']:.6f}"
    )
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

    accelerator.wait_for_everyone()
    save_final_model(accelerator, model, MODEL_PATH, args.output_model_path, args.speaker_name)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
