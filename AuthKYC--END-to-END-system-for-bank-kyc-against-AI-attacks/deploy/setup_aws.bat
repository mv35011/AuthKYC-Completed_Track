@echo off
REM ═══════════════════════════════════════════════════════════
REM AuthKYC — AWS Infrastructure Setup (Windows)
REM ═══════════════════════════════════════════════════════════
REM Prerequisites:
REM   1. AWS CLI installed and configured (aws configure)
REM   2. Docker Desktop installed and running
REM   3. ONNX model exported (deploy/lambda/model/ftca_model.onnx)
REM
REM This script creates:
REM   - S3 bucket for video uploads (authkyc-uploads)
REM   - S3 bucket for results/evidence (authkyc-results)
REM   - ECR repository for the Lambda container
REM   - Lambda function (authkyc-inference)
REM   - API Gateway (for sync invocation from frontend)
REM   - CloudWatch warmup rule (every 5 minutes)
REM   - IAM role for Lambda
REM ═══════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

REM ─── CONFIGURATION (EDIT THESE) ───
set AWS_REGION=ap-south-1
set AWS_ACCOUNT_ID=YOUR_ACCOUNT_ID
set PROJECT_NAME=authkyc
set UPLOAD_BUCKET=%PROJECT_NAME%-uploads
set RESULTS_BUCKET=%PROJECT_NAME%-results
set ECR_REPO=%PROJECT_NAME%-inference
set LAMBDA_NAME=%PROJECT_NAME%-inference
set LAMBDA_MEMORY=3072
set LAMBDA_TIMEOUT=60

echo.
echo ═══════════════════════════════════════════════════════
echo   AuthKYC AWS Setup
echo   Region: %AWS_REGION%
echo ═══════════════════════════════════════════════════════

REM ─── 1. CREATE S3 BUCKETS ───
echo.
echo [1/7] Creating S3 buckets...

aws s3 mb s3://%UPLOAD_BUCKET% --region %AWS_REGION% 2>nul
aws s3 mb s3://%RESULTS_BUCKET% --region %AWS_REGION% 2>nul

REM Enable CORS on upload bucket (for presigned URL uploads from browser)
echo {"CORSRules":[{"AllowedOrigins":["*"],"AllowedMethods":["PUT","GET"],"AllowedHeaders":["*"],"MaxAgeSeconds":3600}]} > cors.json
aws s3api put-bucket-cors --bucket %UPLOAD_BUCKET% --cors-configuration file://cors.json
del cors.json

echo   ✓ S3 buckets created

REM ─── 2. CREATE IAM ROLE ───
echo.
echo [2/7] Creating IAM role...

echo {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]} > trust-policy.json

aws iam create-role --role-name %LAMBDA_NAME%-role --assume-role-policy-document file://trust-policy.json --region %AWS_REGION% 2>nul
del trust-policy.json

REM Attach policies
aws iam attach-role-policy --role-name %LAMBDA_NAME%-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam attach-role-policy --role-name %LAMBDA_NAME%-role --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

echo   ✓ IAM role created
echo   Waiting 10s for IAM propagation...
timeout /t 10 /nobreak >nul

REM ─── 3. CREATE ECR REPOSITORY ───
echo.
echo [3/7] Creating ECR repository...

aws ecr create-repository --repository-name %ECR_REPO% --region %AWS_REGION% 2>nul

REM Login to ECR
aws ecr get-login-password --region %AWS_REGION% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com

echo   ✓ ECR repository created

REM ─── 4. BUILD AND PUSH DOCKER IMAGE ───
echo.
echo [4/7] Building Lambda container image...

REM Copy modules into the Lambda build context
xcopy /E /I /Y ..\..\modules deploy\lambda\modules\ >nul

cd deploy\lambda

docker build -t %ECR_REPO%:latest .
docker tag %ECR_REPO%:latest %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPO%:latest
docker push %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPO%:latest

cd ..\..

echo   ✓ Container image pushed to ECR

REM ─── 5. CREATE LAMBDA FUNCTION ───
echo.
echo [5/7] Creating Lambda function...

aws lambda create-function ^
    --function-name %LAMBDA_NAME% ^
    --package-type Image ^
    --code ImageUri=%AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPO%:latest ^
    --role arn:aws:iam::%AWS_ACCOUNT_ID%:role/%LAMBDA_NAME%-role ^
    --memory-size %LAMBDA_MEMORY% ^
    --timeout %LAMBDA_TIMEOUT% ^
    --environment "Variables={UPLOAD_BUCKET=%UPLOAD_BUCKET%,RESULTS_BUCKET=%RESULTS_BUCKET%,ONNX_MODEL_PATH=/var/task/model/ftca_model.onnx}" ^
    --region %AWS_REGION% 2>nul

REM If function already exists, update it
aws lambda update-function-code ^
    --function-name %LAMBDA_NAME% ^
    --image-uri %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPO%:latest ^
    --region %AWS_REGION% 2>nul

echo   ✓ Lambda function created (%LAMBDA_MEMORY%MB RAM, %LAMBDA_TIMEOUT%s timeout)

REM ─── 6. CREATE FUNCTION URL (simpler than API Gateway) ───
echo.
echo [6/7] Creating Lambda Function URL...

aws lambda create-function-url-config ^
    --function-name %LAMBDA_NAME% ^
    --auth-type NONE ^
    --cors "AllowOrigins=*,AllowMethods=POST,AllowHeaders=Content-Type" ^
    --region %AWS_REGION% 2>nul

REM Add resource policy for public access
aws lambda add-permission ^
    --function-name %LAMBDA_NAME% ^
    --statement-id FunctionURLPublicAccess ^
    --action lambda:InvokeFunctionUrl ^
    --principal "*" ^
    --function-url-auth-type NONE ^
    --region %AWS_REGION% 2>nul

REM Get the function URL
for /f "tokens=*" %%a in ('aws lambda get-function-url-config --function-name %LAMBDA_NAME% --region %AWS_REGION% --query "FunctionUrl" --output text') do set FUNCTION_URL=%%a

echo   ✓ Function URL: %FUNCTION_URL%

REM ─── 7. CLOUDWATCH WARMUP RULE ───
echo.
echo [7/7] Creating CloudWatch warmup rule...

aws events put-rule ^
    --name %LAMBDA_NAME%-warmup ^
    --schedule-expression "rate(5 minutes)" ^
    --state ENABLED ^
    --region %AWS_REGION% 2>nul

REM Get Lambda ARN
for /f "tokens=*" %%a in ('aws lambda get-function --function-name %LAMBDA_NAME% --region %AWS_REGION% --query "Configuration.FunctionArn" --output text') do set LAMBDA_ARN=%%a

aws events put-targets ^
    --rule %LAMBDA_NAME%-warmup ^
    --targets "Id=warmup,Arn=%LAMBDA_ARN%,Input={\"warmup\":true}" ^
    --region %AWS_REGION% 2>nul

aws lambda add-permission ^
    --function-name %LAMBDA_NAME% ^
    --statement-id CloudWatchWarmup ^
    --action lambda:InvokeFunction ^
    --principal events.amazonaws.com ^
    --source-arn arn:aws:events:%AWS_REGION%:%AWS_ACCOUNT_ID%:rule/%LAMBDA_NAME%-warmup ^
    --region %AWS_REGION% 2>nul

echo   ✓ CloudWatch warmup rule: every 5 minutes

REM ─── SUMMARY ───
echo.
echo ═══════════════════════════════════════════════════════
echo   SETUP COMPLETE
echo.
echo   Lambda URL:     %FUNCTION_URL%
echo   Upload Bucket:  s3://%UPLOAD_BUCKET%
echo   Results Bucket: s3://%RESULTS_BUCKET%
echo   Region:         %AWS_REGION% (Mumbai)
echo.
echo   Test with:
echo     aws lambda invoke --function-name %LAMBDA_NAME% --payload "{\"warmup\":true}" response.json
echo.
echo   Disable warmup after demo:
echo     aws events disable-rule --name %LAMBDA_NAME%-warmup --region %AWS_REGION%
echo ═══════════════════════════════════════════════════════

endlocal
