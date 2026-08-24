"""
AuthKYC — AWS Lambda Handler
==============================
Entry point for the Lambda function. Handles:
1. S3 event trigger (video uploaded → process)
2. Direct API Gateway invocation (sync mode)
3. Warmup pings (CloudWatch keepalive)

Environment Variables:
    RESULTS_BUCKET  — S3 bucket for evidence & results (default: authkyc-results)
    ONNX_MODEL_PATH — Path to ONNX model inside container (default: /var/task/model/ftca_model.onnx)
"""
import json
import os
import time
import tempfile
import traceback
import boto3

# Initialize engine at module level (reused across warm invocations)
# This is the key cold-start optimization — engine + model load once.
ENGINE = None
S3_CLIENT = None


def get_engine():
    """Lazy-init the inference engine (loads ONNX model on first call)."""
    global ENGINE
    if ENGINE is None:
        # Add the modules directory to Python path
        import sys
        sys.path.insert(0, '/var/task')
        sys.path.insert(0, '/var/task/modules')

        from inference_engine import LambdaKYCEngine
        onnx_path = os.environ.get('ONNX_MODEL_PATH', '/var/task/model/ftca_model.onnx')
        ENGINE = LambdaKYCEngine(onnx_model_path=onnx_path)
    return ENGINE


def get_s3():
    """Lazy-init S3 client."""
    global S3_CLIENT
    if S3_CLIENT is None:
        S3_CLIENT = boto3.client('s3', region_name=os.environ.get('AWS_REGION', 'ap-south-1'))
    return S3_CLIENT


def handler(event, context):
    """Main Lambda entry point.

    Supports three invocation modes:
    1. Warmup ping: {"warmup": true} → return immediately
    2. S3 trigger: S3 event notification → download, process, upload results
    3. API Gateway: {"video_url": "s3://...", "request_id": "..."} → process and return
    """

    # ─── Mode 1: CloudWatch Warmup Ping ───
    if event.get('warmup') or event.get('source') == 'aws.events':
        # Load the model on warmup so it's cached for real requests
        get_engine()
        return {
            'statusCode': 200,
            'body': json.dumps({'status': 'warm', 'message': 'Engine ready'})
        }

    try:
        start_time = time.time()
        s3 = get_s3()
        engine = get_engine()

        # ─── Mode 2: S3 Event Trigger ───
        if 'Records' in event:
            record = event['Records'][0]
            source_bucket = record['s3']['bucket']['name']
            source_key = record['s3']['object']['key']
            request_id = os.path.splitext(os.path.basename(source_key))[0]

        # ─── Mode 3: Direct API Gateway Invocation ───
        elif 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
            source_bucket = body.get('bucket', os.environ.get('UPLOAD_BUCKET', 'authkyc-uploads'))
            source_key = body['video_key']
            request_id = body.get('request_id', f"req_{int(time.time())}")

        # ─── Mode 3b: Direct invocation (testing) ───
        else:
            source_bucket = event.get('bucket', os.environ.get('UPLOAD_BUCKET', 'authkyc-uploads'))
            source_key = event['video_key']
            request_id = event.get('request_id', f"req_{int(time.time())}")

        print(f"[Lambda] Processing: s3://{source_bucket}/{source_key}")
        print(f"[Lambda] Request ID: {request_id}")

        # Download video to /tmp
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir='/tmp') as tmp:
            tmp_path = tmp.name

        s3.download_file(source_bucket, source_key, tmp_path)
        download_time = time.time() - start_time
        print(f"[Lambda] Downloaded in {download_time:.1f}s")

        # Run 4-stage PAD pipeline
        inference_start = time.time()
        results, evidence = engine.analyze_video_with_evidence(tmp_path)
        inference_time = time.time() - inference_start
        print(f"[Lambda] Inference in {inference_time:.1f}s")

        # Determine final decision
        if results['is_virtual_camera']:
            final_decision = "DENIED: VIRTUAL_CAMERA_INJECTION"
        elif results['is_replay_attack']:
            final_decision = "DENIED: SCREEN_REPLAY_ATTACK"
        elif not results['is_lively']:
            final_decision = "DENIED: BIOLOGICAL_LIVENESS_FAILED"
        elif results['is_deepfake']:
            final_decision = "DENIED: SYNTHETIC_AI_GENERATION"
        else:
            final_decision = "APPROVED"

        # Build response
        response = {
            'request_id': request_id,
            'processing_time_seconds': round(time.time() - start_time, 2),
            'download_time_seconds': round(download_time, 2),
            'inference_time_seconds': round(inference_time, 2),
            'final_decision': final_decision,
            'stages': {
                'stage_1_prnu': {
                    'score': results['prnu_energy'],
                    'passed': not results['is_virtual_camera'],
                    'detail': 'PRNU sensor fingerprint analysis'
                },
                'stage_2_moire': {
                    'score': results['moire_score'],
                    'passed': not results['is_replay_attack'],
                    'detail': 'High-frequency Moiré pattern detection'
                },
                'stage_3_rppg': {
                    'score': results['rppg_snr'],
                    'passed': results['is_lively'],
                    'detail': f"CHROM rPPG biological pulse ({results['biological_bpm']:.1f} BPM)"
                },
                'stage_4_ftca': {
                    'score': results['ai_manipulation_score'],
                    'passed': not results['is_deepfake'],
                    'detail': 'FTCA Frequency-Temporal Cross-Attention'
                }
            }
        }

        # Upload results to S3
        results_bucket = os.environ.get('RESULTS_BUCKET', 'authkyc-results')

        # Upload JSON result
        s3.put_object(
            Bucket=results_bucket,
            Key=f"results/{request_id}.json",
            Body=json.dumps(response, indent=2),
            ContentType='application/json'
        )

        # Upload evidence (FFT spectrum, etc.)
        for name, data in evidence.items():
            s3.put_object(
                Bucket=results_bucket,
                Key=f"evidence/{request_id}/{name}.png",
                Body=data,
                ContentType='image/png'
            )

        print(f"[Lambda] Results uploaded to s3://{results_bucket}/results/{request_id}.json")

        # Cleanup
        os.unlink(tmp_path)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response)
        }

    except Exception as e:
        traceback.print_exc()
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        }
