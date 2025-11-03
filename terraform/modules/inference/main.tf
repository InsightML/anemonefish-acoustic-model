terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Data source for current AWS account and region
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  
  common_tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
    },
    var.tags
  )
  
  # S3 bucket names (must be globally unique)
  input_bucket_name  = "${local.name_prefix}-input-${data.aws_caller_identity.current.account_id}"
  output_bucket_name = "${local.name_prefix}-output-${data.aws_caller_identity.current.account_id}"
}

# ============================================================================
# S3 BUCKETS
# ============================================================================

# Input bucket for temporary audio file storage
resource "aws_s3_bucket" "input" {
  bucket = local.input_bucket_name

  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "input" {
  bucket = aws_s3_bucket.input.id
  versioning_configuration {
    status = "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "input" {
  bucket = aws_s3_bucket.input.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "input" {
  bucket = aws_s3_bucket.input.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets  = true
}

resource "aws_s3_bucket_lifecycle_configuration" "input" {
  bucket = aws_s3_bucket.input.id
  count  = var.s3_lifecycle_enabled && var.s3_input_expiration_days > 0 ? 1 : 0

  rule {
    id     = "delete_old_files"
    status = "Enabled"

    expiration {
      days = var.s3_input_expiration_days
    }
  }
}

# Output bucket for inference results
resource "aws_s3_bucket" "output" {
  bucket = local.output_bucket_name

  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "output" {
  bucket = aws_s3_bucket.output.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "output" {
  bucket = aws_s3_bucket.output.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "output" {
  bucket = aws_s3_bucket.output.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets  = true
}

resource "aws_s3_bucket_lifecycle_configuration" "output" {
  bucket = aws_s3_bucket.output.id
  count  = var.s3_lifecycle_enabled && var.s3_output_expiration_days > 0 ? 1 : 0

  rule {
    id     = "delete_old_results"
    status = "Enabled"

    expiration {
      days = var.s3_output_expiration_days
    }
  }
}

# ============================================================================
# IAM ROLES AND POLICIES
# ============================================================================

# Lambda execution role
resource "aws_iam_role" "lambda_execution" {
  name = "${local.name_prefix}-lambda-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

# Lambda execution policy (S3 access, CloudWatch logs)
resource "aws_iam_role_policy" "lambda_execution" {
  name = "${local.name_prefix}-lambda-execution-policy"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${local.name_prefix}-*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.input.arn}/*",
          "${aws_s3_bucket.output.arn}/*",
          "arn:aws:s3:::${var.model_s3_bucket}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.input.arn,
          aws_s3_bucket.output.arn,
          "arn:aws:s3:::${var.model_s3_bucket}"
        ]
      }
    ]
  })
}

# ============================================================================
# LAMBDA FUNCTION
# ============================================================================

# CloudWatch Log Group for Lambda
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${local.name_prefix}-inference"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# Lambda function with container image
resource "aws_lambda_function" "inference" {
  function_name = "${local.name_prefix}-inference"
  
  package_type = "Image"
  image_uri    = var.lambda_image_uri

  timeout     = var.lambda_timeout
  memory_size = var.lambda_memory_size

  role = aws_iam_role.lambda_execution.arn

  reserved_concurrent_executions = var.lambda_reserved_concurrent_executions

  environment {
    variables = {
      # AWS Configuration
      S3_MODEL_BUCKET  = var.model_s3_bucket
      S3_MODEL_KEY     = var.model_s3_key
      S3_INPUT_BUCKET  = aws_s3_bucket.input.id
      S3_OUTPUT_BUCKET = aws_s3_bucket.output.id

      # Model Configuration
      MODEL_VERSION = var.model_version
      MODEL_CLASSES = var.inference_config.model_classes != null ? var.inference_config.model_classes : "noise,anemonefish,biological"

      # Spectrogram Configuration
      FMAX_HZ       = tostring(var.inference_config.fmax_hz != null ? var.inference_config.fmax_hz : 2000)
      N_FFT         = tostring(var.inference_config.n_fft != null ? var.inference_config.n_fft : 1024)
      HOP_LENGTH    = tostring(var.inference_config.hop_length != null ? var.inference_config.hop_length : 256)
      WIDTH_PIXELS  = tostring(var.inference_config.width_pixels != null ? var.inference_config.width_pixels : 256)
      HEIGHT_PIXELS = tostring(var.inference_config.height_pixels != null ? var.inference_config.height_pixels : 256)

      # Window Configuration
      WINDOW_DURATION = tostring(var.inference_config.window_duration != null ? var.inference_config.window_duration : 1.0)
      STRIDE_DURATION = tostring(var.inference_config.stride_duration != null ? var.inference_config.stride_duration : 0.4)

      # Prediction Configuration
      BATCH_SIZE             = tostring(var.inference_config.batch_size != null ? var.inference_config.batch_size : 32)
      CONFIDENCE_THRESHOLD   = tostring(var.inference_config.confidence_threshold != null ? var.inference_config.confidence_threshold : 0.5)
      TARGET_CLASS           = var.inference_config.target_class != null ? var.inference_config.target_class : "anemonefish"
      MIN_EVENT_DURATION     = tostring(var.inference_config.min_event_duration != null ? var.inference_config.min_event_duration : 0.2)
      MIN_GAP_DURATION       = tostring(var.inference_config.min_gap_duration != null ? var.inference_config.min_gap_duration : 0.1)
      SMOOTHING_WINDOW       = tostring(var.inference_config.smoothing_window != null ? var.inference_config.smoothing_window : 5)
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.lambda,
    aws_iam_role_policy.lambda_execution
  ]

  tags = local.common_tags
}

# ============================================================================
# API GATEWAY
# ============================================================================

# REST API
resource "aws_apigateway_rest_api" "inference_api" {
  name        = "${local.name_prefix}-api"
  description = "API Gateway for anemonefish audio inference"

  endpoint_configuration {
    types = ["REGIONAL"]
  }

  binary_media_types = [
    "multipart/form-data",
    "application/octet-stream",
    "audio/*"
  ]

  tags = local.common_tags
}

# CORS Configuration
resource "aws_apigateway_gateway_response" "cors" {
  rest_api_id   = aws_apigateway_rest_api.inference_api.id
  response_type = "DEFAULT_4XX"

  response_templates = {
    "application/json" = jsonencode({
      message = "$context.error.message"
    })
  }

  response_parameters = {
    "gatewayresponse.header.Access-Control-Allow-Origin"  = join(",", var.cors_allowed_origins)
    "gatewayresponse.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
    "gatewayresponse.header.Access-Control-Allow-Methods" = "'GET,OPTIONS,POST'"
  }
}

# API Key (optional)
resource "aws_apigateway_api_key" "inference_api_key" {
  count = var.enable_api_key ? 1 : 0
  
  name = "${local.name_prefix}-api-key"

  tags = local.common_tags
}

# Usage Plan (optional)
resource "aws_apigateway_usage_plan" "inference_usage_plan" {
  count = var.enable_api_key ? 1 : 0

  name        = "${local.name_prefix}-usage-plan"
  description = "Usage plan for inference API"

  api_stages {
    api_id = aws_apigateway_rest_api.inference_api.id
    stage  = aws_apigateway_stage.inference_api.stage_name
  }

  tags = local.common_tags
}

resource "aws_apigateway_usage_plan_key" "inference_usage_plan_key" {
  count = var.enable_api_key ? 1 : 0

  key_id        = aws_apigateway_api_key.inference_api_key[0].id
  key_type      = "API_KEY"
  usage_plan_id = aws_apigateway_usage_plan.inference_usage_plan[0].id
}

# Lambda Permission for API Gateway
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.inference.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigateway_rest_api.inference_api.execution_arn}/*/*"
}

# /predict resource
resource "aws_apigateway_resource" "predict" {
  rest_api_id = aws_apigateway_rest_api.inference_api.id
  parent_id   = aws_apigateway_rest_api.inference_api.root_resource_id
  path_part   = "predict"
}

# OPTIONS method for CORS preflight
resource "aws_apigateway_method" "predict_options" {
  rest_api_id   = aws_apigateway_rest_api.inference_api.id
  resource_id   = aws_apigateway_resource.predict.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_apigateway_integration" "predict_options" {
  rest_api_id = aws_apigateway_rest_api.inference_api.id
  resource_id = aws_apigateway_resource.predict.id
  http_method = aws_apigateway_method.predict_options.http_method
  type        = "MOCK"

  request_templates = {
    "application/json" = jsonencode({
      statusCode = 200
    })
  }
}

resource "aws_apigateway_method_response" "predict_options" {
  rest_api_id = aws_apigateway_rest_api.inference_api.id
  resource_id = aws_apigateway_resource.predict.id
  http_method = aws_apigateway_method.predict_options.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }

  response_models = {
    "application/json" = "Empty"
  }
}

resource "aws_apigateway_integration_response" "predict_options" {
  rest_api_id = aws_apigateway_rest_api.inference_api.id
  resource_id = aws_apigateway_resource.predict.id
  http_method = aws_apigateway_method.predict_options.http_method
  status_code = aws_apigateway_method_response.predict_options.status_code

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
    "method.response.header.Access-Control-Allow-Methods" = "'GET,OPTIONS,POST'"
    "method.response.header.Access-Control-Allow-Origin"  = join(",", var.cors_allowed_origins)
  }

  depends_on = [aws_apigateway_integration.predict_options]
}

# POST method for /predict
resource "aws_apigateway_method" "predict_post" {
  rest_api_id   = aws_apigateway_rest_api.inference_api.id
  resource_id   = aws_apigateway_resource.predict.id
  http_method   = "POST"
  authorization = var.enable_api_key ? "API_KEY" : "NONE"
}

resource "aws_apigateway_integration" "predict_post" {
  rest_api_id = aws_apigateway_rest_api.inference_api.id
  resource_id = aws_apigateway_resource.predict.id
  http_method = aws_apigateway_method.predict_post.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.inference.invoke_arn
}

resource "aws_apigateway_method_response" "predict_post" {
  rest_api_id = aws_apigateway_rest_api.inference_api.id
  resource_id = aws_apigateway_resource.predict.id
  http_method = aws_apigateway_method.predict_post.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = true
  }

  response_models = {
    "application/json" = "Empty"
  }
}

resource "aws_apigateway_integration_response" "predict_post" {
  rest_api_id = aws_apigateway_rest_api.inference_api.id
  resource_id = aws_apigateway_resource.predict.id
  http_method = aws_apigateway_method.predict_post.http_method
  status_code = aws_apigateway_method_response.predict_post.status_code

  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = "'*'"
  }

  depends_on = [aws_apigateway_integration.predict_post]
}

# API Gateway Deployment
resource "aws_apigateway_deployment" "inference_api" {
  depends_on = [
    aws_apigateway_method.predict_post,
    aws_apigateway_integration.predict_post,
    aws_apigateway_method.predict_options,
    aws_apigateway_integration.predict_options
  ]

  rest_api_id = aws_apigateway_rest_api.inference_api.id

  triggers = {
    redeployment = sha1(jsonencode([
      aws_apigateway_resource.predict.id,
      aws_apigateway_method.predict_post.id,
      aws_apigateway_method.predict_options.id
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }
}

# API Gateway Stage
resource "aws_apigateway_stage" "inference_api" {
  deployment_id = aws_apigateway_deployment.inference_api.id
  rest_api_id   = aws_apigateway_rest_api.inference_api.id
  stage_name    = var.api_gateway_stage_name

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      caller         = "$context.identity.caller"
      user           = "$context.identity.user"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      resourcePath   = "$context.resourcePath"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }

  tags = local.common_tags
}

# CloudWatch Log Group for API Gateway
resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/${local.name_prefix}"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}
