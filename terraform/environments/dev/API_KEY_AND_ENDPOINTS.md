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

Example curl command:
```bash
curl -X POST \
  https://u6ugwtk4gl.execute-api.eu-west-2.amazonaws.com/api/predict \
  -H "x-api-key: VzmBR5Hvnk9jBEpJedfBF5N5hWpRBX6L8IrqRW3M" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@your_audio_file.wav"
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

