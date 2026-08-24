"""
AuthKYC — Railway Backend (API Gateway)
========================================
Thin FastAPI proxy that sits between the frontend and AWS Lambda/S3.
Handles presigned URL generation, Lambda invocation, and result polling.

Deploy on Railway (free tier: 500 hours/month).

Environment Variables (set in Railway dashboard):
    AWS_ACCESS_KEY_ID     — AWS credentials
    AWS_SECRET_ACCESS_KEY — AWS credentials
    AWS_REGION            — default: ap-south-1
    UPLOAD_BUCKET         — default: authkyc-uploads
    RESULTS_BUCKET        — default: authkyc-results
    LAMBDA_NAME           — default: authkyc-inference
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import boto3
import uuid
import os
import json
import time

app = FastAPI(title="AuthKYC Defensive KYC Gateway", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS clients (lazy-init)
_s3_client = None
_lambda_client = None

AWS_REGION = os.environ.get('AWS_REGION', 'ap-south-1')
UPLOAD_BUCKET = os.environ.get('UPLOAD_BUCKET', 'authkyc-uploads')
RESULTS_BUCKET = os.environ.get('RESULTS_BUCKET', 'authkyc-results')
LAMBDA_NAME = os.environ.get('LAMBDA_NAME', 'authkyc-inference')


def get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3', region_name=AWS_REGION)
    return _s3_client


def get_lambda():
    global _lambda_client
    if _lambda_client is None:
        _lambda_client = boto3.client('lambda', region_name=AWS_REGION)
    return _lambda_client


# ── Frontend Serving ──
@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")

# Mount static assets
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/styles.css")
async def serve_css():
    return FileResponse("frontend/styles.css")

@app.get("/app.js")
async def serve_js():
    return FileResponse("frontend/app.js")


# ── API Endpoints ──

class UploadRequest(BaseModel):
    filename: str


class UploadResponse(BaseModel):
    upload_url: str
    video_key: str
    request_id: str


@app.post("/api/v1/get_upload_url", response_model=UploadResponse)
async def get_upload_url(req: UploadRequest):
    """Generate a presigned S3 URL for direct browser upload.

    The video goes directly to S3 — our server never touches raw PII.
    This is the 'presigned URL' pattern for serverless file uploads.
    """
    request_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(req.filename)[1] or '.mp4'
    video_key = f"uploads/{request_id}{ext}"

    s3 = get_s3()
    presigned_url = s3.generate_presigned_url(
        'put_object',
        Params={
            'Bucket': UPLOAD_BUCKET,
            'Key': video_key,
            'ContentType': 'video/mp4'
        },
        ExpiresIn=300  # 5 minutes
    )

    return UploadResponse(
        upload_url=presigned_url,
        video_key=video_key,
        request_id=request_id
    )


class AnalyzeRequest(BaseModel):
    video_key: str
    request_id: str


@app.post("/api/v1/analyze")
async def analyze_video(req: AnalyzeRequest):
    """Invoke Lambda synchronously to analyze the uploaded video.

    Flow: Frontend → Railway → Lambda (sync) → Railway → Frontend
    Total expected time: ~8-12 seconds
    """
    lam = get_lambda()

    payload = {
        'bucket': UPLOAD_BUCKET,
        'video_key': req.video_key,
        'request_id': req.request_id
    }

    try:
        response = lam.invoke(
            FunctionName=LAMBDA_NAME,
            InvocationType='RequestResponse',  # Synchronous
            Payload=json.dumps(payload)
        )

        result = json.loads(response['Payload'].read().decode('utf-8'))

        if response.get('StatusCode') == 200 and 'body' in result:
            body = json.loads(result['body']) if isinstance(result['body'], str) else result['body']
            return JSONResponse(content=body)
        else:
            raise HTTPException(status_code=500, detail=f"Lambda error: {result}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/results/{request_id}")
async def get_results(request_id: str):
    """Fetch results from S3 (for async polling mode)."""
    s3 = get_s3()

    try:
        response = s3.get_object(
            Bucket=RESULTS_BUCKET,
            Key=f"results/{request_id}.json"
        )
        result = json.loads(response['Body'].read().decode('utf-8'))
        return JSONResponse(content=result)
    except s3.exceptions.NoSuchKey:
        return JSONResponse(
            status_code=202,
            content={"status": "processing", "message": "Analysis in progress..."}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/evidence/{request_id}/{filename}")
async def get_evidence(request_id: str, filename: str):
    """Fetch evidence images (FFT spectrum, etc.) from S3."""
    s3 = get_s3()

    try:
        response = s3.get_object(
            Bucket=RESULTS_BUCKET,
            Key=f"evidence/{request_id}/{filename}"
        )
        content = response['Body'].read()
        content_type = response.get('ContentType', 'image/png')

        from fastapi.responses import Response
        return Response(content=content, media_type=content_type)
    except Exception:
        raise HTTPException(status_code=404, detail="Evidence not found")


@app.get("/api/v1/health")
async def health_check():
    """Health check for Railway monitoring."""
    return {
        "status": "healthy",
        "region": AWS_REGION,
        "upload_bucket": UPLOAD_BUCKET,
        "lambda_function": LAMBDA_NAME,
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
