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

output "ecr_repository_name" {
  description = "ECR repository name"
  value       = module.ecr.repository_name
}

output "api_key_id" {
  description = "API Gateway API Key ID (if enabled)"
  value       = module.inference.api_key_id
}

output "api_key_value" {
  description = "API Gateway API Key value (SENSITIVE - store securely)"
  value       = module.inference.api_key_value
  sensitive   = true
}

# Uncomment if using model_bucket module
# output "model_bucket_name" {
#   description = "Model artifacts bucket name"
#   value       = module.model_bucket.bucket_name
# }
