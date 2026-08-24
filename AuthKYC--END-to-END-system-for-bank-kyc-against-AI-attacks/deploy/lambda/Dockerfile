FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies for OpenCV and MediaPipe
RUN yum install -y \
    mesa-libGL \
    mesa-libGLU \
    libSM \
    libXext \
    libXrender \
    && yum clean all

# Install Python dependencies (ONNX Runtime, NOT PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the original modules (PRNU, Moiré, rPPG — all pure numpy/scipy/mediapipe)
COPY ../../modules/ ${LAMBDA_TASK_ROOT}/modules/

# Copy Lambda-specific code
COPY face_cropper.py ${LAMBDA_TASK_ROOT}/
COPY inference_engine.py ${LAMBDA_TASK_ROOT}/
COPY handler.py ${LAMBDA_TASK_ROOT}/

# Copy ONNX model (built by deploy/export_onnx.py after training)
COPY model/ ${LAMBDA_TASK_ROOT}/model/

# Set the handler
CMD ["handler.handler"]
