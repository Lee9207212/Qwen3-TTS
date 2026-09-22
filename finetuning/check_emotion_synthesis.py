# coding=utf-8
"""Synthesize the same sentences under every emotion and measure whether the
emotion vectors actually change the delivery.

The loss curves cannot answer this: on a corpus where the text already implies
the emotion, a model with dead emotion vectors and a model with working ones
score almost the same teacher-forcing CE. The only decisive test is to hold the
text fixed, vary the emotion vector, and look at the audio that comes out.

Decoding samples with a fixed seed re-applied before every call, so the six
emotions are compared under matched randomness while still producing audio the
model is actually good at. Greedy decoding would be more deterministic but a
codec LM tends to degenerate under it, and prosody measured off degenerate
audio is meaningless. Pass --greedy to compare the two.

Usage:
    python3 finetuning/check_emotion_synthesis.py output/final_model --speaker F2 --out wavs/

Needs: librosa, soundfile, numpy (plus qwen-tts itself).
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

EMOTIONS = ["anger", "disgust", "fear", "happy", "sad", "surprise"]

# Deliberately emotion-neutral: nothing in the wording implies a mood, so any
# difference in the output has to come from the emotion vector.
DEFAULT_SENTENCES = [
    "明日の会議は午後三時からです。",
    "駅前の書店で本を三冊買いました。",
    "今週の天気は曇りのち晴れだそうです。",
    "資料は机の上に置いてあります。",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Fine-tuned model directory (e.g. output/final_model)")
    parser.add_argument("--speaker", default="F2", help="--speaker_name used during training")
    parser.add_argument("--out", default="emotion_probe", help="Directory for the generated wavs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="Japanese")
    parser.add_argument("--greedy", action="store_true", help="Also generate a greedy set for comparison")
    # A range this wide invites octave errors. These defaults suit a female
    # speaker; for a male one use roughly --fmin 70 --fmax 250.
    parser.add_argument("--fmin", type=float, default=120.0)
    parser.add_argument("--fmax", type=float, default=350.0)
    return parser.parse_args()


def prosody(wav, sr, text, fmin, fmax):
    """Four numbers that describe delivery rather than content."""
    import librosa

    wav = np.asarray(wav, dtype=np.float32)
    duration = len(wav) / sr
    f0, voiced, _ = librosa.pyin(wav, fmin=fmin, fmax=fmax, sr=sr)
    voiced_f0 = f0[~np.isnan(f0)]
    if voiced_f0.size < 5:
        return None
    # Speech sits near 40-70% voiced. Far below that means the tracker lost the
    # signal, and every pitch number below it is noise.
    voiced_ratio = float(voiced_f0.size) / float(f0.size)
    # Pitch in semitones so the spread is comparable across speakers.
    semitones = 12.0 * np.log2(voiced_f0 / np.median(voiced_f0))
    rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-9)
    return {
        "f0_median": float(np.median(voiced_f0)),
        "f0_spread": float(np.percentile(semitones, 90) - np.percentile(semitones, 10)),
        "energy_db": float(20.0 * np.log10(rms)),
        "chars_per_sec": len(text) / duration if duration > 0 else 0.0,
        "voiced_ratio": voiced_ratio,
        "duration": duration,
    }


FEATURES = [
    ("f0_median", "音高中位數 (Hz)"),
    ("f0_spread", "音高起伏 (半音)"),
    ("energy_db", "能量 (dB)"),
    ("chars_per_sec", "語速 (字/秒)"),
    ("voiced_ratio", "有聲比例 (健檢)"),
]


def report(rows, label):
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    for key, title in FEATURES:
        print(f"\n{title}")
        header = "  emotion    " + "".join(f"{i:>10}" for i in range(len(rows[EMOTIONS[0]])))
        print(header + f"{'mean':>11}")
        for emotion in EMOTIONS:
            values = [entry[key] for entry in rows[emotion]]
            cells = "".join(f"{v:>10.2f}" for v in values)
            print(f"  {emotion:<11}{cells}{np.mean(values):>11.2f}")

    print(f"\n{'-' * 72}\n效果量：情緒造成的變異 / 句子造成的變異\n{'-' * 72}")
    print("  > 1.0 表示換情緒比換句子影響更大（情緒向量有在作用）")
    print("  < 1.0 表示情緒向量的影響小於句子本身的差異\n")
    for key, title in FEATURES:
        matrix = np.array([[entry[key] for entry in rows[emotion]] for emotion in EMOTIONS])
        emotion_means = matrix.mean(axis=1)
        between = float(np.var(emotion_means))
        within = float(np.mean(np.var(matrix - emotion_means[:, None], axis=1)))
        ratio = between / within if within > 1e-12 else float("inf")
        verdict = "情緒主導" if ratio > 1.0 else "句子主導"
        print(f"  {title:<20} {ratio:>8.2f}   {verdict}")


def main():
    args = parse_args()
    try:
        import librosa  # noqa: F401
    except ImportError:
        raise SystemExit("This script needs librosa: pip install librosa")

    import torch
    from qwen_tts import Qwen3TTSModel

    os.makedirs(args.out, exist_ok=True)
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path, device_map=args.device, dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    supported = tts.get_supported_speakers() or []
    wanted = [f"{args.speaker}_{emotion}" for emotion in EMOTIONS]
    folded = {name.casefold() for name in supported}
    missing = [name for name in wanted if name.casefold() not in folded]
    if missing:
        print(f"Speakers present in the checkpoint: {supported}", file=sys.stderr)
        raise SystemExit(f"Missing speakers: {missing}. Check --speaker matches --speaker_name.")

    passes = [("sampled", {"do_sample": True, "temperature": 0.9, "top_p": 0.9})]
    if args.greedy:
        passes.append(("greedy", {"do_sample": False}))

    for tag, gen_kwargs in passes:
        rows = {emotion: [] for emotion in EMOTIONS}
        for index, text in enumerate(DEFAULT_SENTENCES):
            for emotion in EMOTIONS:
                torch.manual_seed(1234)
                wavs, sr = tts.generate_custom_voice(
                    text=text,
                    speaker=f"{args.speaker}_{emotion}",
                    language=args.language,
                    **gen_kwargs,
                )
                path = os.path.join(args.out, f"{tag}_s{index}_{emotion}.wav")
                sf.write(path, wavs[0], sr)
                measured = prosody(wavs[0], sr, text, args.fmin, args.fmax)
                if measured is None:
                    raise SystemExit(f"Could not measure F0 in {path}; listen to it first.")
                rows[emotion].append(measured)
        report(rows, f"{tag} decoding | {len(DEFAULT_SENTENCES)} sentences x {len(EMOTIONS)} emotions")

    print(f"\nWavs written to {os.path.abspath(args.out)} -- listen to them before trusting any table.")


if __name__ == "__main__":
    main()
