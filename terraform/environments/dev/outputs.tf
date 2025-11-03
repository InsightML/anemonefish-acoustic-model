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
