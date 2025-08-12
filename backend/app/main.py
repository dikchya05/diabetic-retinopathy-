from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import traceback

from .predict import predict, gradcam_base64
from .config import CLASS_NAMES

app = FastAPI(title="DR Inference API")

# Add CORS Middleware
origins = [
    "http://localhost:3000",  # Next.js dev server
    "http://127.0.0.1:3000",  # Alternate localhost
    "https://yourfrontend.com"  # Production domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # Allowed origins
    allow_credentials=True,
    allow_methods=["*"],          # Allow all HTTP methods
    allow_headers=["*"],          # Allow all headers
)

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Upload an image file.")
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        pred_idx, confidence, probs = predict(image)

        try:
            cam_b64 = gradcam_base64(image)
        except Exception:
            cam_b64 = None

        return JSONResponse({
            "prediction": int(pred_idx),
            "label": CLASS_NAMES[int(pred_idx)],
            "confidence": confidence,
            "probabilities": probs,
            "gradcam_base64": cam_b64
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
