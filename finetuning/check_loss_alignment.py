#!/usr/bin/env python3
"""Verify Qwen3-TTS finetuning loss label alignment with toy labels."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch
from transformers.loss.loss_utils import ForCausalLMLoss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_tts.core.models.modeling_qwen3_tts import (  # noqa: E402
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
)

IGNORE_INDEX = -100
EOS = 99
VOCAB = 128


def effective_causal_targets(labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]


def logits_for_targets(targets: torch.Tensor, vocab_size: int = VOCAB) -> torch.Tensor:
    logits = torch.full((*targets.shape, vocab_size), -20.0)
    for index in torch.cartesian_prod(*[torch.arange(s) for s in targets.shape]):
        idx = tuple(int(i) for i in index)
        target = int(targets[idx])
        if target != IGNORE_INDEX:
            logits[idx + (target,)] = 20.0
    return logits


def print_mapping(title: str, contexts: list[str], targets: torch.Tensor) -> None:
    print(title)
    for context, target in zip(contexts, targets.tolist()[0]):
        print(f"  {context} -> {target}")


def assert_source_uses_shift_labels() -> None:
    source = inspect.getsource(Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward_finetune)
    assert "shift_labels=labels.contiguous()" in source
    assert "labels=None" in source


def main() -> int:
    assert_source_uses_shift_labels()

    full_labels = torch.tensor([[10, 20, 30, 40, EOS]])
    contexts = ["10", "20", "30", "40", "EOS"]
    main_targets = effective_causal_targets(full_labels)
    main_logits = logits_for_targets(main_targets)
    main_loss = ForCausalLMLoss(main_logits, full_labels, vocab_size=VOCAB)
    print_mapping("main correct path", contexts, main_targets)
    assert main_targets.tolist()[0][:4] == [20, 30, 40, EOS]
    assert float(main_loss) < 1e-4

    preshifted_labels = full_labels[:, 1:]
    wrong_targets = effective_causal_targets(preshifted_labels)
    print_mapping("main old double-shift path", contexts[:4], wrong_targets)
    assert wrong_targets.tolist()[0][:3] == [30, 40, EOS]

    sub_targets = torch.tensor([[20, 30, 40, 50]])
    sub_contexts = ["hidden/main", "codebook0", "codebook1", "codebook2"]
    sub_logits = logits_for_targets(sub_targets)
    sub_loss = ForCausalLMLoss(
        sub_logits,
        labels=None,
        shift_labels=sub_targets.contiguous(),
        vocab_size=VOCAB,
    )
    print_mapping("sub correct path", sub_contexts, sub_targets)
    assert sub_targets.tolist()[0] == [20, 30, 40, 50]
    assert float(sub_loss) < 1e-4

    wrong_sub_targets = effective_causal_targets(sub_targets)
    print_mapping("sub old double-shift path", sub_contexts, wrong_sub_targets)
    assert wrong_sub_targets.tolist()[0][:3] == [30, 40, 50]

    print("alignment_check=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
