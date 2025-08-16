# OCR/HTR — Handwritten Text Recognition + Text-to-Speech (TTS)

End-to-end pipeline to **recognize handwritten text** from images and (optionally) **speak the result**.  
Baseline model: **CRNN (CNN + BiLSTM + CTC)**. Inference via **CLI** and **FastAPI**. TTS via **pyttsx3 (offline)** or **gTTS (online)**.  
**Status:** TTS is implemented and usable from both CLI and API.

---

## ✨ Features
- Line-level HTR with **CTC decoding** (greedy / beam).
- Preprocessing: grayscale → normalize → height-preserving resize → width padding.
- **Config-first** (YAML) training/eval; deterministic seeds.
- Metrics: **CER** (Character Error Rate), **WER** (Word Error Rate).
- Model export: PyTorch `.pt` and optional **ONNX**.
- **Text-to-Speech**: toggle from CLI/API; choose `pyttsx3` or `gTTS`.

---

## 🧰 Requirements & Install
- Python **3.10+**
- PyTorch **2.x** (CPU or CUDA)
- FastAPI + Uvicorn
- TTS: `pyttsx3` (offline) or `gTTS` (online)

```bash
# virtual env (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# install
pip install -r requirements.txt
# or minimal:
pip install torch torchvision torchaudio            # add correct CUDA index URL if needed
pip install fastapi uvicorn[standard] pyyaml numpy opencv-python pillow jiwer onnx onnxruntime
pip install pyttsx3 gTTS
```

Linux (offline TTS) may require:
```bash
sudo apt-get update && sudo apt-get install -y espeak libespeak1
```

---

## 📦 Dataset
Use any line-level handwriting dataset (e.g., **IAM**). Respect the dataset license—do not redistribute.

Preprocess example:
```bash
python -m src.utils.data_utils   --dataset iam   --raw_dir data/raw/iam   --out_dir data/processed/iam   --img_height 32 --max_width 512 --normalize true
```

This creates manifests like `train_labels.txt` with lines:
```
path/to/img.png \t target text here
```

---

## ⚙️ Config (YAML)
`src/configs/base.yaml` example:
```yaml
seed: 1337
data:
  train_manifest: data/processed/iam/train_labels.txt
  val_manifest:   data/processed/iam/val_labels.txt
  img_height: 32
  max_width: 512
model:
  num_classes: 96
  cnn_out: 256
  lstm_hidden: 256
train:
  epochs: 50
  batch_size: 64
  lr: 0.0005
  num_workers: 4
  amp: true
decode:
  mode: greedy          # greedy | beam
  beam_size: 10
```

---

## 🏃 Training
```bash
python src/train.py --config src/configs/base.yaml --out models/crnn_base
# resume
python src/train.py --config src/configs/base.yaml --out models/crnn_base --resume models/crnn_base/last.pt
```

## ✅ Evaluation
```bash
python src/evaluate.py   --config src/configs/base.yaml   --checkpoint models/crnn_base/best.pt   --manifest data/processed/iam/val_labels.txt
```
Prints overall **CER/WER** plus sample predictions.

---

## 🔎 Inference — CLI
Single image:
```bash
python src/infer.py   --checkpoint models/crnn_base/best.pt   --image samples/line.png   --speak off
```

Batch of images (folder):
```bash
python src/infer.py   --checkpoint models/crnn_base/best.pt   --dir samples/lines   --out predictions.csv   --speak on --tts_engine pyttsx3
```

### 🗣️ Text-to-Speech (implemented)
- `--speak on` enables speech after recognition.
- Choose engine with `--tts_engine pyttsx3` (offline) or `--tts_engine gtts` (online).
- Minimal wrapper idea used by the project:
```python
def speak(text: str, engine: str = "pyttsx3"):
    if engine == "gtts":
        from gtts import gTTS
        import tempfile, webbrowser
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        gTTS(text).save(tmp.name)
        webbrowser.open(tmp.name)  # simple cross-platform playback
    else:
        import pyttsx3
        tts = pyttsx3.init()
        tts.say(text)
        tts.runAndWait()
```

---

## 🌐 Inference — API (FastAPI)
Start the server:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` → `{ "status": "ok" }`
- `POST /recognize?speak=true&engine=pyttsx3`
  - `multipart/form-data`: `file=@line.png`
  - or JSON: `{ "image_b64": "..." }`

cURL:
```bash
curl -X POST "http://localhost:8000/recognize?speak=true&engine=pyttsx3"   -F "file=@samples/line.png"
```

Example response:
```json
{ "text": "recognized text", "cer": 0.12, "timing_ms": 34 }
```

---

## 🗜️ ONNX Export (optional)
```bash
python - <<'PY'
import torch, onnx
from src.models.crnn import CRNN
m = CRNN(num_classes=96)
m.load_state_dict(torch.load("models/crnn_base/best.pt", map_location="cpu"))
m.eval()
dummy = torch.randn(1, 1, 32, 512)
torch.onnx.export(
    m, dummy, "models/crnn_base/model.onnx",
    input_names=["image"], output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}
)
PY
```

---

## 🧪 Accuracy Tips
- Fix height (e.g., 32 px), preserve aspect ratio, pad to a max width.
- Augment lightly (small rotations, affine, elastic distortions).
- Try beam search (`beam_size: 10–20`).
- Increase LSTM hidden or add attention for tougher datasets.
- Clean labels; CER is sensitive to noise.

---

## 🔒 License & Data
- Code: pick **MIT** or **Apache-2.0** unless you need copyleft.
- Datasets (e.g., **IAM**) have separate licenses; users must obtain them directly.

---

## 🙌 Acknowledgements
- CRNN + CTC literature and open baselines.
- IAM Handwriting Database (for line images/transcriptions).

---

## 🧾 Changelog
- **v0.1** — CRNN + CTC baseline; greedy/beam decode; CLI + FastAPI; **TTS implemented** (pyttsx3/gTTS); optional ONNX export.
