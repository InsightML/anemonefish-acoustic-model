output "lambda_function_name" {
  description = "Name of the Lambda function"
  value       = aws_lambda_function.inference.function_name
}

output "lambda_function_arn" {
  description = "ARN of the Lambda function"
  value       = aws_lambda_function.inference.arn
}

output "lambda_function_invoke_arn" {
  description = "Invoke ARN of the Lambda function"
  value       = aws_lambda_function.inference.invoke_arn
}

output "api_gateway_id" {
  description = "ID of the API Gateway REST API"
  value       = aws_api_gateway_rest_api.inference_api.id
}

output "api_gateway_url" {
  description = "URL of the API Gateway endpoint"
  value       = "https://${aws_api_gateway_rest_api.inference_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.inference_api.stage_name}"
}

output "api_gateway_predict_endpoint" {
  description = "Full URL of the /predict endpoint"
  value       = "${aws_api_gateway_rest_api.inference_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.inference_api.stage_name}/predict"
}

output "api_gateway_predict_url" {
  description = "Full HTTPS URL of the /predict endpoint"
  value       = "https://${aws_api_gateway_rest_api.inference_api.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.inference_api.stage_name}/predict"
}

output "s3_input_bucket_name" {
  description = "Name of the S3 input bucket"
  value       = aws_s3_bucket.input.id
}

output "s3_input_bucket_arn" {
  description = "ARN of the S3 input bucket"
  value       = aws_s3_bucket.input.arn
}

output "s3_output_bucket_name" {
  description = "Name of the S3 output bucket"
  value       = aws_s3_bucket.output.id
}

output "s3_output_bucket_arn" {
  description = "ARN of the S3 output bucket"
  value       = aws_s3_bucket.output.arn
}

output "lambda_execution_role_arn" {
  description = "ARN of the Lambda execution role"
  value       = aws_iam_role.lambda_execution.arn
}

output "api_key_id" {
  description = "API Key ID (if enabled)"
  value       = var.enable_api_key ? aws_api_gateway_api_key.inference_api_key[0].id : null
}

output "api_key_value" {
  description = "API Key value (if enabled). Store this securely."
  value       = var.enable_api_key ? aws_api_gateway_api_key.inference_api_key[0].value : null
  sensitive   = true
}

output "cloudwatch_log_group_lambda" {
  description = "CloudWatch log group for Lambda"
  value       = aws_cloudwatch_log_group.lambda.name
}

output "cloudwatch_log_group_api_gateway" {
  description = "CloudWatch log group for API Gateway"
  value       = aws_cloudwatch_log_group.api_gateway.name
}
