# AWS Infrastructure Deployment - Dev Environment

## API Endpoint
```
https://u6ugwtk4gl.execute-api.eu-west-2.amazonaws.com/api/predict
```

## API Key (KEEP SECURE!)
```
VzmBR5Hvnk9jBEpJedfBF5N5hWpRBX6L8IrqRW3M
```

### How to use the API Key
Add this header to your API requests:
```
x-api-key: VzmBR5Hvnk9jBEpJedfBF5N5hWpRBX6L8IrqRW3M
```

## ⚠️ IMPORTANT: File Size Limitation

**API Gateway has a 10MB payload limit for synchronous requests.**

### For Small Files (< 10MB):
```bash
curl -X POST \
  https://u6ugwtk4gl.execute-api.eu-west-2.amazonaws.com/api/predict \
  -H "x-api-key: VzmBR5Hvnk9jBEpJedfBF5N5hWpRBX6L8IrqRW3M" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@your_small_audio_file.wav"
```

### For Large Files (> 10MB) - RECOMMENDED APPROACH:

**Step 1: Upload directly to S3**
```bash
# Upload your file to the input bucket
aws s3 cp your_audio_file.wav s3://anemonefish-inference-dev-input-944269089535/
```

**Step 2: Invoke Lambda directly with S3 path** ✅ TESTED & WORKING
```bash
aws lambda invoke \
  --function-name anemonefish-inference-dev-inference \
  --cli-binary-format raw-in-base64-out \
  --payload '{"s3_bucket": "anemonefish-inference-dev-input-944269089535", "s3_key": "your_audio_file.wav"}' \
  --region eu-west-2 \
  --cli-read-timeout 0 \
  response.json
```

**Step 3: Check results in output bucket**
```bash
aws s3 ls s3://anemonefish-inference-dev-output-944269089535/
aws s3 cp s3://anemonefish-inference-dev-output-944269089535/results.json ./
```

## Other Resources

- **Lambda Function**: `anemonefish-inference-dev-inference`
- **ECR Repository**: `944269089535.dkr.ecr.eu-west-2.amazonaws.com/anemonefish-inference-dev`
- **S3 Input Bucket**: `anemonefish-inference-dev-input-944269089535`
- **S3 Output Bucket**: `anemonefish-inference-dev-output-944269089535`

## Next Steps

1. **Build and Push Docker Image**: Before the Lambda can work, you need to build and push the inference Docker image to ECR
2. **Test the API**: Once the Docker image is pushed, test the API endpoint
3. **Monitor Logs**: Check CloudWatch logs at:
   - `/aws/lambda/anemonefish-inference-dev-inference`
   - `/aws/apigateway/anemonefish-inference-dev`

## Security Note

⚠️ **IMPORTANT**: Keep this API key secure! Anyone with this key can use your API.
- Do not commit this file to version control
- Store it in a secure password manager
- Rotate the key regularly
- Consider implementing rate limiting

## Regenerating the API Key

If you need to regenerate the key:
```bash
cd /Volumes/InsightML/NAS/3_Lucia_Yllan/Clown_Fish_Acoustics/terraform/environments/dev
terraform taint module.inference.aws_api_gateway_api_key.inference_api_key[0]
terraform apply -auto-approve
terraform output -raw api_key_value
```

## Future Improvements for Production

To handle large files properly via API Gateway, consider implementing:

1. **S3 Presigned URLs**: Generate presigned upload URLs so clients upload directly to S3
2. **Async Processing**: Use SQS to queue processing jobs
3. **Webhook Callbacks**: Notify clients when processing completes
4. **API Gateway HTTP API**: Supports larger payloads (up to 10MB body + headers)
5. **Chunked Upload**: Split large files into chunks for upload

Example presigned URL flow:
```
1. Client requests upload URL → API returns presigned S3 URL
2. Client uploads file directly to S3 using presigned URL
3. S3 event triggers Lambda for processing
4. Results saved to output bucket
5. Optional: SNS notification or webhook callback to client
```

