output "api_endpoint" {
  description = "API Gateway endpoint URL"
  value       = module.inference.api_gateway_predict_url
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = module.inference.lambda_function_name
}

output "s3_input_bucket" {
  description = "S3 input bucket name"
  value       = module.inference.s3_input_bucket_name
}

output "s3_output_bucket" {
  description = "S3 output bucket name"
  value       = module.inference.s3_output_bucket_name
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = module.ecr.repository_url
}

output "api_key_id" {
  description = "API Key ID (if enabled)"
  value       = module.inference.api_key_id
}

output "api_key_value" {
  description = "API Key value (if enabled). Store this securely."
  value       = module.inference.api_key_value
  sensitive   = true
}
