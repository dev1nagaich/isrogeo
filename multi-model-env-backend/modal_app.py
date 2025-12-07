import modal
import os
import sys
from modal import asgi_app
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # On Modal, python-dotenv might not be installed; that's fine.
    pass

app = modal.App("multi-model-env-backend")

# --------- ACTIVATE ZIP ENV ----------
def activate_env(env_dir): 
    # Add /root to sys.path for src/ module imports
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    
    for folder in os.listdir(env_dir):
        candidate = os.path.join(env_dir, folder)
        if os.path.isdir(candidate):
            lib_path = os.path.join(candidate, "lib")
            if os.path.exists(lib_path):
                for pyfolder in os.listdir(lib_path):
                    # Match python3.10, python3.11, python3.12 etc. (must have 2-digit minor version)
                    if pyfolder.startswith("python3.") and pyfolder[9:10].isdigit():
                        site_packages = os.path.join(lib_path, pyfolder, "site-packages")
                        if os.path.exists(site_packages):
                            sys.path.insert(0, site_packages)
                            print(f"✅ Activated: {site_packages}")
                            return site_packages
    raise RuntimeError(f"❌ Could not locate site-packages inside env: {env_dir}")

# --------- BACKEND IMAGE (FASTAPI) ----------
backend_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1",
        "libglib2.0-0"
    )
    .pip_install(
        "annotated-doc==0.0.4",
        "annotated-types==0.7.0",
        "anyio==4.11.0",
        "bcrypt==4.0.1",
        "click==8.3.1",
        "colorama==0.4.6",
        "dnspython==2.8.0",
        "email-validator==2.3.0",
        "fastapi==0.121.3",
        "h11==0.16.0",
        "idna==3.11",
        "motor==3.7.1",
        "passlib[bcrypt]==1.7.4",
        "pyasn1==0.6.1",
        "pycparser==2.23",
        "pydantic==2.12.4",
        "pydantic_core==2.41.5",
        "PyJWT==2.10.1",
        "pymongo==4.15.4",
        "python-dotenv==1.2.1",
        "python-multipart==0.0.20",
        "six==1.17.0",
        "sniffio==1.3.1",
        "starlette==0.50.0",
        "typing_extensions==4.15.0",
        "typing-inspection==0.4.2",
        "uvicorn==0.38.0",
        "openai",
        "requests",
        "pillow",
        "numpy<2",
        "torch",
        "transformers",
        "peft",
        "opencv-python",
        "einops",
        "timm"
    )
    .add_local_dir("src", remote_path="/root/src")
)



# MongoDB connection secret
mongodb_secret = modal.Secret.from_name("mongodb-secret")

# JWT secret
jwt_secret = modal.Secret.from_name("jwt-secret")


# -------- ZIP MODEL ENV IMAGES ---------
caption_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("unzip", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.57.1",
        "accelerate==1.0.1",
        "pillow==11.3.0",
        "numpy==1.26.4",
        "peft==0.18.0",
        "einops==0.8.1",
        "timm==1.0.22",
        "opencv-python-headless==4.10.0.84",
        "safetensors==0.6.2",
    )
    .run_commands("mkdir -p /captioning_env", "mkdir -p /root/checkpoints")
    .add_local_file("captioning.zip", "/captioning.zip", copy=True)
    .run_commands("unzip /captioning.zip -d /captioning_env")
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .add_local_dir("checkpoints/captioning_model", remote_path="/root/checkpoints/captioning_model", copy=True)
)




vqa_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("unzip", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "transformers==4.57.1",
        "accelerate==1.11.0",
        "pillow==11.3.0",
        "numpy==1.24.3",
        "peft==0.18.0",
        "einops==0.8.1",
        "timm==1.0.22",
        "safetensors==0.6.2",
    )
    .run_commands("mkdir -p /vqa_env", "mkdir -p /root/checkpoints")
    .add_local_file("vqa.zip", "/vqa.zip", copy=True)
    .run_commands("unzip /vqa.zip -d /vqa_env")
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .add_local_dir("checkpoints/vqa_model", remote_path="/root/checkpoints/vqa_model", copy=True)
)

grounding_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04")  # Use devel image for nvcc
    .apt_install(
        "software-properties-common",
        "unzip",
        "libgl1",
        "libglib2.0-0",
        "git"
    )
    .run_commands(
        # Install system Python so Modal can detect a Python version
        "apt-get update",
        "apt-get install -y python3 python3-venv python3-pip",
        "ln -sf /usr/bin/python3 /usr/bin/python",
        "python --version"
    )

    # 1) Add your zipped conda env
    .add_local_file("grounding.zip", "/grounding.zip", copy=True)

    # 2) Unzip into /grounding_env
    .run_commands(
        "mkdir -p /grounding_env /root/checkpoints",
        "unzip /grounding.zip -d /grounding_env"
    )

    # 3) Fix missing __init__.py files in LLaVA package (grounding.zip has broken LLaVA)
    .run_commands(
        # Create missing __init__.py files in llava subdirectories
        "touch /grounding_env/ground/lib/python3.10/site-packages/llava/model/language_model/__init__.py",
        "touch /grounding_env/ground/lib/python3.10/site-packages/llava/model/multimodal_encoder/__init__.py",
        "touch /grounding_env/ground/lib/python3.10/site-packages/llava/model/multimodal_projector/__init__.py",
        # Fix the model/__init__.py to show actual import errors instead of silently passing
        "echo 'dHJ5OgogICAgZnJvbSAubGFuZ3VhZ2VfbW9kZWwubGxhdmFfbGxhbWEgaW1wb3J0IExsYXZhTGxhbWFGb3JDYXVzYWxMTSwgTGxhdmFDb25maWcKICAgIGZyb20gLmxhbmd1YWdlX21vZGVsLmxsYXZhX21wdCBpbXBvcnQgTGxhdmFNcHRGb3JDYXVzYWxMTSwgTGxhdmFNcHRDb25maWcKICAgIGZyb20gLmxhbmd1YWdlX21vZGVsLmxsYXZhX21pc3RyYWwgaW1wb3J0IExsYXZhTWlzdHJhbEZvckNhdXNhbExNLCBMbGF2YU1pc3RyYWxDb25maWcKZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgaW1wb3J0IHRyYWNlYmFjawogICAgcHJpbnQoZiJMTGFWQSBtb2RlbCBpbXBvcnQgZXJyb3I6IHtlfSIpCiAgICB0cmFjZWJhY2sucHJpbnRfZXhjKCkKICAgIHJhaXNlCg==' | base64 -d > /grounding_env/ground/lib/python3.10/site-packages/llava/model/__init__.py",
        # Install missing packages required by transformers/pydantic - typing_extensions needs Sentinel support
        "/grounding_env/ground/bin/python -m pip install regex safetensors 'typing_extensions>=4.12.0'",
        # Verify the fix works
        "/grounding_env/ground/bin/python -c 'from llava.model.builder import load_pretrained_model; print(\"LLaVA import OK\")'"
    )

    # 4) Add your code + weights
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .add_local_file("checkpoints/best.pt", remote_path="/root/checkpoints/best.pt", copy=True)
    .add_local_dir(
        "checkpoints/llava-v1.5-7b-task-geoground",
        remote_path="/root/checkpoints/llava-v1.5-7b-task-geoground",
        copy=True
    )
)


@app.function(
    image=caption_image,
    gpu="A100-40GB",
    timeout=300,
    memory=4096
)
def run_caption(image_bytes: bytes, max_tokens: int = 512, temperature: float = 0.7):
    """
    Generate caption using Florence-2 Caption Service
    
    Args:
        image_bytes: Image file as bytes
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        {"caption": "Generated caption text"}
    """
    import tempfile
    print(f"📝 Caption Service - Processing image ({len(image_bytes)} bytes)")
    
    # Activate caption environment
    activate_env("/captioning_env")
    
    # Import service from activated environment
    from src.services.florence2_caption_service import get_caption_service
    
    try:
        # Save image bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        # Get caption service (singleton) with correct Modal path
        service = get_caption_service(model_path="/root/checkpoints/captioning_model")
        
        # Generate caption
        caption = service.generate_caption(
            image=tmp_path,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        print(f"✅ Caption generated: {caption[:100]}...")
        return {"caption": caption}
        
    except Exception as e:
        print(f"❌ Caption error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "caption": "Failed to generate caption"}
    finally:
        # Cleanup
        try:
            import os
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except:
            pass


@app.function(
    image=vqa_image,
    gpu="A100-40GB",
    timeout=300,
    memory=4096
)
def run_vqa(image_bytes: bytes, question: str, max_tokens: int = 128, temperature: float = 0.7):
    """
    Answer question using Florence-2 VQA Service
    
    Args:
        image_bytes: Image file as bytes
        question: Question to answer
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        {"answer": "Answer text"}
    """
    import tempfile
    print(f"❓ VQA Service - Question: {question}")
    
    # Activate VQA environment
    activate_env("/vqa_env")
    
    # Import service from activated environment
    from src.services.florence2_vqa_service import get_vqa_service
    
    try:
        # Save image bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        # Get VQA service (singleton) with correct Modal path
        service = get_vqa_service(model_path="/root/checkpoints/vqa_model")
        
        # Answer question
        answer = service.answer_question(
            image=tmp_path,
            question=question,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        print(f"✅ Answer: {answer}")
        return {"answer": answer}
        
    except Exception as e:
        print(f"❌ VQA error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "answer": "Failed to answer question"}
    finally:
        # Cleanup
        try:
            import os
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except:
            pass


@app.function(
    image=grounding_image,
    gpu="A100-40GB",
    timeout=600,
    memory=16384
)
def run_grounding(image_bytes: bytes, query: str, max_boxes: int = 10):
    """
    Detect objects using GeoGround Service
    
    Args:
        image_bytes: Image file as bytes
        query: Natural language query
        max_boxes: Maximum boxes to return
        
    Returns:
        {"detections": [{"object_id": "1", "obbox": [...]}]}
    """
    import tempfile
    import sys
    
    # Remove Modal's injected deps path which has old typing_extensions
    modal_deps = [p for p in sys.path if '/__modal/deps' in p]
    for p in modal_deps:
        sys.path.remove(p)
    
    # Remove any cached typing_extensions before we import the correct one
    typing_ext_modules = [m for m in sys.modules if m.startswith('typing_extensions')]
    for m in typing_ext_modules:
        del sys.modules[m]
    
    print(f"🎯 Grounding Service - Query: {query}")
    
    # Activate grounding environment
    activate_env("/grounding_env")
    
    # Import service from activated environment
    from src.services.grounding_service import get_grounding_service
    
    try:
        # Save image bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        # Get grounding service (singleton) with correct Modal paths
        service = get_grounding_service(
            yolo_model_path="/root/checkpoints/best.pt",
            geoground_model_path="/root/checkpoints/llava-v1.5-7b-task-geoground"
        )
        
        # Detect objects - returns List[Tuple[str, List[float]]]
        detections = service.detect_objects(
            image=tmp_path,
            query=query
        )
        
        # Format detections to match API response
        formatted_detections = [
            {
                "object_id": obj_id,
                "obbox": obbox
            }
            for obj_id, obbox in detections
        ]
        
        print(f"✅ Detected {len(formatted_detections)} objects")
        return {"detections": formatted_detections}
        
    except Exception as e:
        print(f"❌ Grounding error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "detections": []}
    finally:
        # Cleanup
        try:
            import os
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except:
            pass


# ========================================================================================
# SIMPLE ROUTER ENDPOINT (For direct service calls)
# ========================================================================================

@app.function(image=backend_image)
@modal.fastapi_endpoint(method="POST")
def router(request: dict):
    """
    Simple router for individual service calls
    
    Request format:
    {
        "service": "caption" | "vqa" | "grounding",
        "image": "path/to/image.jpg",
        "query": "question or query text",  // For VQA and Grounding
        "max_tokens": 512,  // Optional
        "temperature": 0.7,  // Optional
        "max_boxes": 10  // Optional for grounding
    }
    
    Returns:
        Service-specific response
    """
    service = request.get("service")
    image = request.get("image")
    query = request.get("query", "")
    max_tokens = request.get("max_tokens", 512)
    temperature = request.get("temperature", 0.7)
    max_boxes = request.get("max_boxes", 10)
    
    print(f"\n🔀 Router - Service: {service}")
    
    if service == "caption":
        return run_caption.remote(image, max_tokens, temperature)
    
    elif service == "vqa":
        if not query:
            return {"error": "Query/question is required for VQA"}
        return run_vqa.remote(image, query, max_tokens, temperature)
    
    elif service == "grounding":
        if not query:
            return {"error": "Query is required for grounding"}
        return run_grounding.remote(image, query, max_boxes)
    
    else:
        return {
            "error": "Unknown service. Choose: caption / vqa / grounding",
            "available_services": ["caption", "vqa", "grounding"]
        }


# ========================================================================================
# FASTAPI BACKEND (Complete GeoNLI Evaluation)
# ========================================================================================

@app.function(
    image=backend_image,
    secrets=[mongodb_secret, jwt_secret],
    timeout=600,
    memory=2048
)
@modal.asgi_app()
def fastapi_backend():
    """
    Complete FastAPI backend with GeoNLI evaluation endpoint
    
    This hosts the full backend including:
    - Authentication
    - Session management
    - GeoNLI evaluation endpoint
    - All other API endpoints
    """
    print("🚀 Starting FastAPI backend...")
    
    # Set Modal environment flag
    os.environ["MODAL_ENV"] = "true"
    
    # Import FastAPI app
    from src.server import app as fastapi_app
    
    print("✅ FastAPI backend ready")
    return fastapi_app


# ========================================================================================
# MONGODB CONNECTION TEST
# ========================================================================================

@app.function(
    image=backend_image,
    secrets=[mongodb_secret, jwt_secret]
)
def test_mongodb():
    """Test MongoDB connection"""
    import os
    from pymongo import MongoClient
    
    print("\n" + "="*70)
    print("🔍 MONGODB CONNECTION TEST")
    print("="*70)
    
    mongodb_url = os.getenv("MONGODB_URL")
    
    if not mongodb_url:
        return {
            "status": "error",
            "message": "MONGODB_URL environment variable not set",
            "solution": "Create Modal secret: modal secret create mongodb-secret MONGODB_URL='mongodb+srv://...'"
        }
    
    print(f"📍 MongoDB URL: {mongodb_url[:30]}...")
    
    if "localhost" in mongodb_url:
        return {
            "status": "error",
            "message": "MongoDB URL points to localhost!",
            "current_url": mongodb_url,
            "solution": "Update secret with Atlas URL: modal secret create mongodb-secret MONGODB_URL='mongodb+srv://...'"
        }
    
    try:
        print("🔗 Attempting connection...")
        client = MongoClient(mongodb_url, serverSelectionTimeoutMS=10000)
        
        # Test connection
        server_info = client.server_info()
        
        # List databases
        db_list = client.list_database_names()
        
        print("✅ Connection successful!")
        print(f"📊 MongoDB version: {server_info.get('version', 'unknown')}")
        print(f"📚 Databases: {', '.join(db_list[:5])}")
        print("="*70)
        
        return {
            "status": "success",
            "message": "MongoDB connection successful",
            "mongodb_version": server_info.get('version'),
            "databases": db_list,
            "url_preview": mongodb_url[:30] + "..."
        }
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("="*70)
        
        return {
            "status": "error",
            "message": str(e),
            "url_preview": mongodb_url[:30] + "...",
            "troubleshooting": [
                "1. Check MongoDB Atlas Network Access (allow 0.0.0.0/0)",
                "2. Verify username/password in connection string",
                "3. Ensure cluster is not paused",
                "4. Check connection string format"
            ]
        }


# ========================================================================================
# HEALTH CHECK
# ========================================================================================

@app.function(image=backend_image)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint"""
    import torch
    
    return {
        "status": "healthy",
        "service": "GeoNLI Multi-Model Backend",
        "environment": "Modal",
        "cuda_available": torch.cuda.is_available(),
        "services": {
            "caption": "available",
            "vqa": "available",
            "grounding": "available (A100)"
        }
    }


# ========================================================================================
# LOCAL ENTRYPOINT
# ========================================================================================

@app.local_entrypoint()
def main():
    """Local entrypoint for testing"""
    print("=" * 70)
    print("🚀 GeoNLI Multi-Model Backend - Modal Deployment")
    print("=" * 70)
    print()
    print("Available commands:")
    print("  modal run modal_app.py::test_mongodb    - Test MongoDB connection")
    print("  modal deploy modal_app.py                - Deploy to Modal")
    print()
    print("Available services:")
    print("  - Caption Generation (Florence-2)")
    print("  - Visual Q&A (Florence-2)")
    print("  - Object Grounding (GeoGround)")
    print()
    print("=" * 70)