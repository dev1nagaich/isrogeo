# 🛰️ GeoNLI — Satellite Image Intelligence Platform

> **Inter IIT Tech Meet** · Problem Statement Submission  
> **Representing: IIT Mandi** 🏔️

---

## 🏆 About This Project

This repository contains our submission for the **Inter IIT Tech Meet** problem statement on AI-powered geospatial intelligence. The team from **IIT Mandi** built **GeoNLI** — a full-stack platform for natural language querying of satellite imagery, combining state-of-the-art vision-language models with a production-grade cloud deployment pipeline.

---

## 🎯 Problem Statement

Design and implement an end-to-end system capable of interpreting satellite images using natural language. The system must support:

- **Captioning** — Generate detailed scene descriptions from remote sensing imagery
- **Visual Question Answering (VQA)** — Answer natural language questions about image content
- **Object Grounding** — Localize objects with oriented bounding boxes (OBB) given a text query

---

## 🏗️ System Architecture

```
User / Frontend
      │
      ▼
FastAPI Backend (GeoNLI API)
      │
      ├──▶ Caption Service     →  Florence-2 + LoRA fine-tune
      ├──▶ VQA Service         →  Florence-2 + LoRA fine-tune
      └──▶ Grounding Service   →  GeoGround (LLaVA-1.5) + YOLO OBB
                │
                ▼
         Modal Cloud (GPU: A100-40GB)
                │
                ▼
         MongoDB Atlas (Auth, Sessions, Projects)
```

**Request flow:** `User → FastAPI Route → Controller → AI Service → GPU Model → Response`

---

## 🤖 AI Models

| Service | Model | Task | Output |
|---------|-------|------|--------|
| **Caption** | Florence-2-large + LoRA | Scene description | Free-form text |
| **VQA** | Florence-2-large + LoRA | Binary / Numeric / Semantic Q&A | Structured answers |
| **Grounding** | GeoGround (LLaVA-1.5-7B) + YOLO OBB | Object detection | 8-point normalized OBB coordinates |

### Grounding Pipeline (Hybrid)
The grounding service uses a smart model-selection strategy:
1. **YOLO OBB** runs first (fast inference, high precision on trained categories)
2. If YOLO confidence < threshold → falls back to **GeoGround** (LLaVA-based, handles open-vocabulary queries)

OBB format: `[x1, y1, x2, y2, x3, y3, x4, y4]` — all values normalized to `[0, 1]`

---

## 📡 API Reference

### Main Evaluation Endpoint

```http
POST /geoNLI/eval
Content-Type: application/json
```

**Request:**
```json
{
  "input_image": {
    "image_id": "sample1.png",
    "image_url": "https://...",
    "metadata": {
      "width": 512,
      "height": 512,
      "spatial_resolution_m": 1.57
    }
  },
  "queries": {
    "caption_query": {
      "instruction": "Generate a detailed caption describing all visible elements"
    },
    "grounding_query": {
      "instruction": "Locate all storage tanks in the image"
    },
    "attribute_query": {
      "binary":  { "instruction": "Are there any digits visible?" },
      "numeric": { "instruction": "How many storage tanks are visible?" },
      "semantic": { "instruction": "What color are the storage tanks?" }
    }
  }
}
```

**Response:**
```json
{
  "input_image": { ... },
  "queries": {
    "caption_query": {
      "instruction": "...",
      "response": "The satellite image shows an industrial area with..."
    },
    "grounding_query": {
      "instruction": "...",
      "response": [
        { "object-id": "1", "obbox": [0.23, 0.12, 0.27, 0.14, 0.29, 0.18, 0.25, 0.16] }
      ]
    },
    "attribute_query": {
      "binary":  { "instruction": "...", "response": "Yes" },
      "numeric": { "instruction": "...", "response": 8.0 },
      "semantic": { "instruction": "...", "response": "White with dark roofs" }
    }
  }
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI 0.121, Python 3.10/3.11 |
| **AI Framework** | PyTorch, HuggingFace Transformers, PEFT (LoRA) |
| **Object Detection** | Ultralytics YOLO (OBB), GeoGround / LLaVA-1.5 |
| **Database** | MongoDB Atlas (async via Motor) |
| **Authentication** | JWT (PyJWT), bcrypt |
| **Cloud Deployment** | Modal (serverless GPU — A100-40GB) |
| **Image Processing** | Pillow, OpenCV, NumPy |

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone and enter backend directory
cd multi-model-env-backend

# 2. Create virtual environment
python3.11 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (copy and fill in values)
cp .env.example .env

# 5. Start server
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## ☁️ Modal Deployment

```bash
# Install and authenticate
pip install modal
modal token new

# Create secrets
modal secret create mongodb-secret MONGODB_URL="mongodb+srv://..."
modal secret create jwt-secret JWT_SECRET="$(openssl rand -hex 32)"

# Deploy
modal deploy modal_app.py
```

Each model runs in its own isolated Docker image with environment-specific dependencies, activated via a custom `activate_env()` utility that hot-patches `sys.path` inside Modal containers.

---

## 📁 Repository Structure

```
multi-model-env-backend/
├── modal_app.py              # Modal deployment — GPU functions & web endpoints
├── modal_client.py           # Python client for calling Modal endpoints
├── requirements.txt
└── src/
    ├── server.py             # FastAPI app entrypoint
    ├── routes/               # API route definitions
    ├── controllers/          # Business logic & orchestration
    ├── services/             # AI model service wrappers
    │   ├── florence2_caption_service.py
    │   ├── florence2_vqa_service.py
    │   └── grounding_service.py
    ├── models/               # Pydantic request/response schemas
    ├── modals/               # MongoDB document schemas
    ├── lib/                  # DB connection & JWT utilities
    ├── middleware/           # Auth middleware
    └── utils/                # Image download & processing helpers
```

---

## 👥 Team — IIT Mandi

Submitted as part of the **Inter IIT Tech Meet** competition.  
Representing **Indian Institute of Technology Mandi**, Himachal Pradesh.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
