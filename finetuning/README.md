## Fine Tuning Qwen3-TTS-12Hz-1.7B/0.6B-Base

The Qwen3-TTS-12Hz-1.7B/0.6B-Base model series currently supports single-speaker fine-tuning. Please run `pip install qwen-tts` first, then run the command below:

```
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS/finetuning
```

Then follow the steps below to complete the entire fine-tuning workflow. Multi-speaker fine-tuning and other advanced fine-tuning features will be supported in future releases.

### 1) Input JSONL format

Prepare your training file as a JSONL (one JSON object per line). Each line must contain:

- `audio`: path to the target training audio (wav)
- `text`: transcript corresponding to `audio`
- `ref_audio`: path to the reference speaker audio (wav)

Example:
```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav"}
```

`ref_audio` recommendation:
- Strongly recommended: use the same `ref_audio` for all samples.
- Keeping `ref_audio` identical across the dataset usually improves speaker consistency and stability during generation.


### 2) Prepare data (extract `audio_codes`)

Convert `train_raw.jsonl` into a training JSONL that includes `audio_codes`:

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```


### 3) Fine-tune

Run SFT using the prepared JSONL:

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name speaker_test
```

Checkpoints will be written to:
- `output/checkpoint-epoch-0`
- `output/checkpoint-epoch-1`
- `output/checkpoint-epoch-2`
- ...


### 4) Quick inference test

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="She said she would be here by noon.",
    speaker="speaker_test",
)
sf.write("output.wav", wavs[0], sr)
```

### Emotion-conditioned LoRA fine-tuning (`sft_12hz_Lora.py`)

`sft_12hz_Lora.py` trains one speaker with a per-emotion offset, so a single
checkpoint can speak the same voice in several emotions. It does not touch the
inference code: each emotion is registered as its own speaker name.

**How it works.** Position 6 of the prompt is the model's only conditioning
vector. Speaker-only fine-tuning writes the frozen speaker-encoder output there;
this script writes `speaker + emotion[i]` instead, where the emotion table is a
zero-initialised `len(EMOTIONS) x hidden_size` tensor trained alongside the LoRA
adapters. Because an emotion row only participates in the forward pass of its own
samples, every other row receives zero gradient — each row is shaped solely by
its own data.

Generation performs a single embedding lookup and has no notion of addition, so
the sums are precomputed at save time and stored in `codec_embedding` rows
`3000 .. 3000 + len(EMOTIONS) - 1`. For `Qwen3-TTS-12Hz-0.6B-Base` that table has
3072 rows, sampling is restricted to ids below 2048, and the special codec ids
end at 2157, so this range is unreachable by the talker. Verify it for any other
checkpoint before training:

```bash
python check_codec_slots.py --model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base
```

**Extra JSONL field.** Every line needs an `emotion` field matching one of the
names in `EMOTIONS` (see `dataset.py`):

```jsonl
{"audio":"./data/utt0001.wav","text":"...","ref_audio":"./data/ref.wav","emotion":"anger"}
```

Keep `ref_audio` identical across the whole dataset. If the reference audio
varies with emotion, the speaker encoder already encodes the emotion and the
table has nothing left to learn — the rows end up near-identical.

**Training.** The emotion table lives outside the PEFT wrapper (anything added
after `get_peft_model()` is frozen by PEFT or dropped by `merge_and_unload()`),
so it gets its own optimizer group. It is tiny and zero-initialised, and needs a
much larger learning rate than the adapters:

```bash
python sft_12hz_Lora.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path output \
  --dataset_dir ./dataset \
  --speaker_name F2 \
  --lr 2e-5 \
  --emotion_lr 1e-3 \
  --num_epochs 3
```

Split the data so every emotion appears in each split; training aborts if an
emotion is missing from the training split and warns if it is missing from
validation.

**Per-emotion validation.** Support is usually uneven, so an emotion that never
fitted can hide behind the others in a single averaged number. Each epoch the
script also evaluates one emotion at a time and prints a row per emotion plus
the best and worst, and appends the same figures to
`output/emotion_loss_history.csv` in long format (one row per split/emotion).
This is a second sweep of the validation set, not six, because the rows are
partitioned; `--emotion_eval_every N` reports every N epochs and `0` turns it
off. `loss_history.csv` and the early-stopping decision are unaffected -- they
still use the aggregate pass.

**Speaker vector constancy.** `save_final_model` bakes the *first* sample of the
*first* batch as the speaker vector for every exported row, which is only
meaningful while `ref_audio` is one fixed file. The script measures the relative
spread between the speaker vectors -- within a batch and against the first batch
-- warns as soon as it exceeds 5%, and prints the verdict at save time. The
tolerance is there to absorb bf16 rounding; a genuinely different reference
shows up at O(100%). If it warns, the saved voice is an arbitrary pick among the
references seen.

**Checking that it learned anything.** The first training step is numerically
identical to speaker-only fine-tuning, since the table starts at zero. At save
time the script prints each emotion vector's norm (next to the speaker vector's,
as a scale reference) and their pairwise cosine similarity. Similarities near
1.0 across the board mean the table did not differentiate — check the
`ref_audio` and `--emotion_lr` first. The raw table is also written to
`output/emotion_table.pt`.

**Inference.** Speaker names are `<speaker_name>_<emotion>`:

```python
wavs, sr = tts.generate_custom_voice(
    text="...",
    speaker="F2_anger",
)
```


### Giving the emotion table a bigger share of the gradient

The adapters and the emotion table are two parameter groups under one loss, so
they compete for the same error signal. The adapters see the transcript and
have far more capacity, so on a corpus whose text already implies the emotion
they explain most of it first, and the table only ever learns the residual.
Two flags address that without touching the data.

**`--freeze_lora_epochs N`** holds the adapters at `requires_grad=False` for
the first N epochs, so the emotion table trains alone and has first claim on
the error signal. Backward still reaches the table through the frozen weights;
the adapters simply stop accumulating gradients and AdamW skips them. This also
tests the competition explanation directly: if the emotion effect grows, the
table was being crowded out; if nothing moves, it has hit its capacity and the
conditioning channel itself needs to change.

**`--select_by emotion_delta`** changes which checkpoint is kept. Validation
loss cannot see whether emotion conditioning works -- when the transcript
implies the emotion, a dead emotion table and a working one score nearly the
same, so selecting on it is close to selecting at random with respect to the
thing being trained. Every epoch now also evaluates the validation set with the
emotion offsets zeroed and logs the gap as `val_emotion_delta` in
`loss_history.csv`. A larger gap means the emotion vectors are carrying more of
the prediction. `--select_by emotion_delta` keeps the epoch that maximises it;
the default `val_loss` preserves the old behaviour. The column is written
either way, so a plain run still shows where the emotion effect peaks -- and
that epoch is usually not the one with the lowest validation loss.

Only `main_loss` enters the delta. The sub-talker term sits on a plateau
throughout training and would add noise to the difference.

```bash
python3 Qwen3-TTS/finetuning/sft_12hz_Lora.py   --dataset_dir <data> --speaker_name F2   --freeze_lora_epochs 5 --select_by emotion_delta
```

### Diagnosing a run where the emotion vectors do not seem to do anything

`loss_history.csv` cannot tell you whether emotion conditioning worked. On a
corpus where the text already implies the emotion, a model with dead emotion
vectors and one with working vectors score almost the same teacher-forcing
cross-entropy, because both can read the emotion off the transcript. Two
scripts cover what the loss curves miss.

**`check_text_leakage.py` -- is the emotion already in the text?**

```bash
python3 finetuning/check_text_leakage.py <train.jsonl> <val.jsonl>
```

Fits a character-n-gram classifier on the transcripts alone (no tokenizer
needed, so it works for Japanese) and reports how well the emotion label can be
recovered from text without hearing any audio. Accuracy far above chance means
the emotion vectors are competing against a free and much stronger signal: the
model can minimise the loss by reading the words, so little gradient is left
for the table. It also prints the most emotion-predictive n-grams, which is
usually enough to see which words are leaking.

Corpora whose scripts were written per emotion -- JVNV and JTES among them --
leak heavily by construction, and the leak cannot be cleaned away, because the
transcript has to match what was actually spoken.

**`check_emotion_synthesis.py` -- do the vectors change the audio?**

```bash
python3 finetuning/check_emotion_synthesis.py output/final_model --speaker F2
```

Synthesises the same emotion-neutral sentences under every emotion and compares
pitch, pitch range, energy and speaking rate. Decoding is greedy by default so
that differences between the six outputs come from the emotion vector rather
than from sampling noise; `--sample` adds a sampled pass for listening.

Alongside the per-emotion table it reports an effect size for each feature:
the variance explained by the emotion divided by the variance explained by the
choice of sentence. Below 1.0 means swapping the emotion matters less than
swapping the sentence, which is the quantitative form of "the vectors are not
doing anything". Listen to the wavs regardless -- the table is corroboration,
not the verdict.

### One-click shell script example

```bash
#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"

RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="output"

BATCH_SIZE=2
LR=2e-5
EPOCHS=3
SPEAKER_NAME="speaker_1"

python prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

python sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME}
```