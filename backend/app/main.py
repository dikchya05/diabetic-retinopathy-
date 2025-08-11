from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import traceback

from .predict import predict, gradcam_base64
from .config import CLASS_NAMES

app = FastAPI(title="DR Inference API")

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
        except Exception as e:
            cam_b64 = None

        return JSONResponse({
            "prediction": int(pred_idx),
            "label": CLASS_NAMES[int(pred_idx)],
            "confidence": confidence,
            "probabilities": probs,
            "gradcam_base64": cam_b64
        })

    except Exception as e:
        traceback.print_exc()  # Prints error stack trace to your server console
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
