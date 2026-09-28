<p align="center">
  <h1 align="center">🎬 FrameScribe</h1>
  <p align="center">
    <strong>Video-to-Text Intelligence Pipeline</strong>
    <br />
    Extract speech, identify speakers, detect scenes — all locally on your machine.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/whisper-large--v3-green?logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/license-MIT-purple" />
</p>

---

## 📖 What is FrameScribe?

**FrameScribe** is a local-first video intelligence tool designed for **content creators**. It takes any video file and extracts:

- 🎤 **Accurate speech transcripts** (English + Hindi, with code-switching support)
- 🗣️ **Speaker identification** (who is speaking when)
- 👁️ **Dense Video Captioning** (1 FPS visual analysis via Qwen-VL)
- 📊 **Structured output** ready for AI tools like Gemini, NotebookLM, and others

The extracted data can be used to generate infographics, B-roll scripts, subtitles, or any downstream content — all without sending your data to the cloud.

### Why FrameScribe?

| Problem | FrameScribe Solution |
|---|---|
| YouTube auto-captions are inaccurate | Whisper large-v3 provides near-human accuracy |
| No context about *what's happening* visually | Qwen2-VL analyzes the video second-by-second to describe actions |
| Can't tell who is speaking | Speaker diarization labels each speaker |
| Data leaves your machine | 100% local processing, your data stays private |
| Expensive cloud transcription APIs | Completely free after setup |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FrameScribe Backend                  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              FastAPI Server (:8000)              │   │
│  │                                                  │   │
│  │  REST API          WebSocket         API Docs    │   │
│  │  /api/*           /ws/progress      /docs        │   │
│  └──────────┬───────────────┬───────────────────────┘   │
│             │               │                           │
│  ┌──────────▼───────────────▼───────────────────────┐   │
│  │            Processing Pipeline                   │   │
│  │                                                  │   │
│  │  ┌─────────┐  ┌───────────┐  ┌───────────────┐   │   │
│  │  │ FFmpeg  │  │  Whisper  │  │ FFmpeg (1 FPS)│   │   │
│  │  │ Audio   │  │ large-v3  │  │ + Qwen2-VL    │   │   │
│  │  │ Extract │  │ STT       │  │ Dense Captions│   │   │
│  │  └────┬────┘  └─────┬─────┘  └──────┬────────┘   │   │
│  │       │             │               │            │   │
│  │       │      ┌──────┴──────┐        │            │   │
│  │       │      │  pyannote   │        │            │   │
│  │       │      │  Speaker    │        │            │   │
│  │       │      │  Diarize    │        │            │   │
│  │       │      └──────┬──────┘        │            │   │
│  │       │             │               │            │   │
│  │  ┌────▼─────────────▼───────────────▼────────┐   │   │
│  │  │          Pipeline Merger                  │   │   │
│  │  │  Combine transcripts + speakers + visuals │   │   │
│  │  └─────────────────┬─────────────────────────┘   │   │
│  │                    │                             │   │
│  │  ┌─────────────────▼─────────────────────────┐   │   │
│  │  │           Formatter / Exporter            │   │   │
│  │  │   JSON │ SRT │ VTT │ TXT │ AI-Ready       │   │   │
│  │  └───────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌───────────────┐                                      │
│  │  SQLite DB    │  Project history & metadata          │
│  └───────────────┘                                      │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 Processing Pipeline

The pipeline processes a video through **4 sequential stages**, with stages 3a and 3b running **in parallel** for efficiency:

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────────────────┐     ┌────────────┐
│   Stage 1    │     │    Stage 2      │     │         Stage 3              │     │  Stage 4   │
│              │     │                 │     │                              │     │            │
│  Audio       │────►│  Transcription  │────►│  ┌─ 3a: Speaker Diarize ─┐  │────►│  Merge &   │
│  Extraction  │     │  (Whisper)      │     │  │   (pyannote-audio)    │  │     │  Enrich    │
│  (FFmpeg)    │     │                 │     │  ├─ 3b: Visual Analysis ─┤  │     │            │
│              │     │                 │     │  │ (Qwen2-VL @ 1 FPS)    │  │     │            │
└──────────────┘     └─────────────────┘     │  └───────────────────────┘  │     └──────┬─────┘
                                             │      (run in parallel)      │            │
                                             └──────────────────────────────┘            ▼
                                                                               Enriched Transcript
                                                                                  (JSON output)
```

### Stage Details

| Stage | Component | Input | Output |
|---|---|---|---|
| **1. Audio Extraction** | FFmpeg | Video file (MP4, MOV, MKV, AVI, WebM) | 16kHz mono WAV |
| **2. Transcription** | faster-whisper (large-v3) | WAV audio | Timestamped text segments with word-level timing |
| **3a. Speaker Diarization** | pyannote-audio 3.1 | WAV audio | Speaker labels with time ranges |
| **3b. Visual Analysis** | Qwen2-VL (2B-Instruct) | 1 FPS extracted images | Semantic natural language descriptions merged into visual timelines |
| **4. Merge & Enrich** | Pipeline Merger | All above outputs | Enriched `TranscriptResult` |

### Why This Architecture?

- **Dense Visual Context**: Instead of just detecting hard cuts, Qwen-VL acts as a narrator, describing the action second-by-second (e.g., "A car driving on a highway").
- **Decoupled services**: Each service (transcriber, diarizer, visual analyzer) is independently testable and replaceable.
- **Parallel execution**: Diarization and scene detection don't depend on each other, so they run simultaneously
- **Lazy loading**: ML models load only on first request, keeping startup fast
- **Async-first**: All I/O-bound operations are async; CPU-bound ML inference runs in thread pools via `asyncio.to_thread()`
- **Graceful degradation**: If diarization isn't available (no HuggingFace token), the pipeline still works — it just labels everything as "Speaker 1"

---

## 📦 Output Format

FrameScribe produces **enriched transcripts** where each text segment includes speaker identity, language, confidence, and visual context:

```json
{
  "video_file": "my_video.mp4",
  "duration_seconds": 342.5,
  "languages": ["en", "hi"],
  "speakers_detected": 2,
  "scenes_detected": 5,
  "created_at": "2026-09-28T00:00:00+00:00",
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 4.52,
      "text": "Hello everyone, welcome to today's video",
      "speaker": "Speaker 1",
      "language": "en",
      "confidence": 0.95,
      "words": [
        { "word": "Hello", "start": 0.0, "end": 0.42, "confidence": 0.98 },
        { "word": "everyone", "start": 0.42, "end": 0.91, "confidence": 0.96 }
      ],
      "visual_context": {
        "scene_id": 1,
        "description": "Scene 1",
        "key_frame": "output/frames/scene_0001.jpg"
      }
    },
    {
      "id": 2,
      "start": 4.52,
      "end": 8.10,
      "text": "आज हम बात करेंगे AI के बारे में",
      "speaker": "Speaker 1",
      "language": "hi",
      "confidence": 0.91,
      "words": [],
      "visual_context": {
        "scene_id": 1,
        "description": "Scene 1",
        "key_frame": "output/frames/scene_0001.jpg"
      }
    }
  ],
  "scenes_summary": [
    { "scene_id": 1, "start": 0.0, "end": 8.10, "key_frame": "output/frames/scene_0001.jpg" },
    { "scene_id": 2, "start": 8.10, "end": 25.0, "key_frame": "output/frames/scene_0002.jpg" }
  ],
  "full_text": "Hello everyone, welcome to today's video. आज हम बात करेंगे AI के बारे में..."
}
```

### Export Formats

| Format | File | Use Case |
|---|---|---|
| **JSON** | `.json` | Structured data for programmatic use, AI tool input |
| **SRT** | `.srt` | Standard subtitles for video editors (Premiere, DaVinci) |
| **VTT** | `.vtt` | Web subtitles (YouTube, HTML5 video) |
| **TXT** | `.txt` | Plain text transcript with speaker labels |
| **AI-Ready** | `_ai.txt` | Formatted with metadata headers for pasting into Gemini / NotebookLM |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Conda** (Miniconda or Anaconda)
- **FFmpeg** installed and on PATH
- **GPU** (recommended): NVIDIA GPU with CUDA, or Apple Silicon (M1/M2/M3/M4)

### Installation

```bash
# 1. Clone / navigate to the project
cd FrameScribe

# 2. Create conda environment
conda create -n videoAI python=3.11 -y
conda activate videoAI

# 3. Install PyTorch (choose one):

# For Apple Silicon (M1/M2/M3/M4):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# For NVIDIA GPU (CUDA 12.x):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
pip install -r backend/requirements.txt
```

### Configuration (Optional)

Create a `.env` file in the project root for optional settings:

```env
# Whisper model size (tiny, base, small, medium, large-v3)
# Smaller = faster but less accurate. Default: large-v3
WHISPER_MODEL_SIZE=large-v3

# HuggingFace token for speaker diarization (optional)
# Get yours at: https://huggingface.co/settings/tokens
# Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
HF_TOKEN=hf_your_token_here
```

### Running the Server

```bash
# Activate environment
conda activate videoAI

# Start the API server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Server will be available at:
#   API:  http://localhost:8000
#   Docs: http://localhost:8000/docs
```

> **Note**: The first transcription will download the Whisper model (~3GB for large-v3). This only happens once — the model is cached locally after that.

---

## 📡 API Reference

### Health Check

```bash
GET /api/health
```

Response:
```json
{
  "status": "healthy",
  "device": "cpu",
  "model": "large-v3",
  "diarization_available": false
}
```

### Transcribe a Video

```bash
POST /api/transcribe
Content-Type: multipart/form-data

# Upload a video file
curl -X POST http://localhost:8000/api/transcribe \
  -F "file=@my_video.mp4"
```

Response:
```json
{ "job_id": "550e8400-e29b-41d4-a716-446655440000" }
```

### Check Job Status

```bash
GET /api/status/{job_id}
```

Response:
```json
{
  "job_id": "550e8400-...",
  "status": "transcribing",
  "progress": 0.45,
  "stage_progress": 0.45,
  "current_stage": "transcribing"
}
```

Status values: `pending` → `extracting_audio` → `transcribing` → `diarizing_and_scenes` → `merging` → `completed` / `failed`

### Get Result

```bash
GET /api/result/{job_id}
```

Returns the full `TranscriptResult` JSON (see Output Format above).

### Real-Time Progress (WebSocket)

```javascript
const ws = new WebSocket("ws://localhost:8000/api/ws/progress/{job_id}");
ws.onmessage = (event) => {
  const status = JSON.parse(event.data);
  console.log(`${status.current_stage}: ${(status.stage_progress * 100).toFixed(0)}%`);
};
```

### Export Transcript

```bash
# Export via POST (provide transcript data)
POST /api/export
Content-Type: application/json

{
  "transcript": { ... },  // TranscriptResult object
  "format": "srt"         // json | srt | vtt | txt | ai_ready
}

# Download directly from a completed job
GET /api/export/download/{job_id}/{format}

# Example:
curl -O http://localhost:8000/api/export/download/550e8400-.../srt
```

### Project History

```bash
# List all projects
GET /api/projects

# Delete a project
DELETE /api/projects/{project_id}
```

### Interactive API Docs

Visit **http://localhost:8000/docs** for the full Swagger UI where you can test all endpoints interactively.

---

## 📁 Project Structure

```
FrameScribe/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Configuration & device detection
│   │
│   ├── models/
│   │   └── schemas.py             # Pydantic data models
│   │
│   ├── services/
│   │   ├── audio_extractor.py     # FFmpeg: video → 16kHz mono WAV
│   │   ├── transcriber.py         # faster-whisper: audio → timestamped text
│   │   ├── diarizer.py            # pyannote: audio → speaker labels
│   │   ├── scene_detector.py      # PySceneDetect: video → scene boundaries + key frames
│   │   ├── formatter.py           # Export: JSON / SRT / VTT / TXT / AI-Ready
│   │   └── pipeline.py            # Orchestrator: ties all services together
│   │
│   ├── routers/
│   │   ├── transcribe.py          # /api/transcribe, /api/status, /api/result, WebSocket
│   │   └── export.py              # /api/export, /api/export/download
│   │
│   ├── db/
│   │   └── database.py            # SQLite async database for project history
│   │
│   └── requirements.txt           # Python dependencies
│
├── .env                           # (Optional) Environment variables
└── README.md                      # This file
```

### Runtime Directories (auto-created)

```
backend/
├── output/
│   ├── audio/                     # Extracted WAV files
│   ├── frames/                    # Key frame JPGs per scene
│   └── exports/                   # Exported transcript files
│
└── data/
    ├── uploads/                   # Uploaded video files
    ├── results/                   # Completed transcript JSONs
    └── videoai.db                 # SQLite database
```

---

## ⚙️ Tech Stack

| Component | Technology | Role |
|---|---|---|
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) | Async REST API + WebSocket |
| **Speech-to-Text** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2-based Whisper (4x faster, 2x less memory) |
| **Speaker Diarization** | [pyannote.audio](https://github.com/pyannote/pyannote-audio) | State-of-the-art speaker identification |
| **Scene Detection** | [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | Content-aware scene cut detection |
| **Video Processing** | [FFmpeg](https://ffmpeg.org/) | Audio extraction, format conversion |
| **Computer Vision** | [OpenCV](https://opencv.org/) | Key frame extraction |
| **Database** | [SQLite](https://www.sqlite.org/) (via aiosqlite) | Async project history |
| **Data Validation** | [Pydantic v2](https://docs.pydantic.dev/) | Request/response models |

### Hardware Support

| Device | Support | Notes |
|---|---|---|
| NVIDIA GPU (CUDA) | ✅ Full | float16 inference, fastest |
| Apple Silicon (M1/M2/M3/M4) | ✅ Full | int8 on CPU (CTranslate2), still fast |
| CPU only | ✅ Full | int8 inference, slower but works |

---

## 🗺️ Roadmap

- [x] Speech transcription (Whisper large-v3)
- [x] Word-level timestamps
- [x] Multi-language support (English + Hindi code-switching)
- [x] Speaker diarization
- [x] Scene detection + key frame extraction
- [x] Multi-format export (JSON, SRT, VTT, TXT, AI-Ready)
- [x] REST API + WebSocket progress
- [x] Project history (SQLite)
- [ ] AI-powered scene descriptions (Florence-2 local / Gemini API)
- [ ] Keyword & topic extraction
- [ ] Direct Gemini integration ("Generate B-Roll Script")
- [ ] NotebookLM-optimized export
- [ ] Electron desktop app with React UI
- [ ] Batch video processing
- [ ] Speaker naming (replace "Speaker 1" → "Aditya")
- [ ] Cloud API fallback (Deepgram, AssemblyAI)

---

## 🤝 Contributing

This is currently a personal project. Feel free to fork, modify, and adapt it for your own workflow.

---

## 📄 License

MIT License — use it however you want.
