terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    # Configure backend with: terraform init -backend-config=backend.hcl
    # Or use local backend for dev: remove backend block
    bucket         = null  # Set your Terraform state bucket
    key            = "anemonefish-inference/dev/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = null  # Set your DynamoDB table for state locking
  }
}

# Data source for AWS account
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# ============================================================================
# ECR REPOSITORY
# ============================================================================

module "ecr" {
  source = "../../modules/ecr"

  project_name = var.project_name
  environment  = "dev"

  image_tag_mutability    = "MUTABLE"
  scan_on_push            = true
  lifecycle_policy_enabled = true
  max_image_count          = 10

  tags = var.tags
}

# ============================================================================
# INFERENCE INFRASTRUCTURE
# ============================================================================

module "inference" {
  source = "../../modules/inference"

  project_name = var.project_name
  environment  = "dev"
  aws_region   = data.aws_region.current.name

  # Model configuration (adjust as needed)
  model_s3_bucket = var.model_s3_bucket
  model_s3_key    = var.model_s3_key
  model_version   = var.model_version

  # Lambda configuration
  lambda_timeout                   = 900  # 15 minutes
  lambda_memory_size              = 3008  # Max memory for large audio processing
  lambda_reserved_concurrent_executions = null  # Unlimited for dev

  # Lambda container image (must be built and pushed to ECR first)
  # Use the ECR repository URL from the module output
  lambda_image_uri = var.lambda_image_uri != null ? var.lambda_image_uri : "${module.ecr.repository_url}:latest"

  # API Gateway configuration
  api_gateway_stage_name = "api"
  cors_allowed_origins   = ["*"]  # Allow all origins in dev
  enable_api_key         = false  # No API key for dev

  # Logging
  log_retention_days = 7  # Shorter retention for dev

  # S3 lifecycle
  s3_lifecycle_enabled      = true
  s3_input_expiration_days  = 3   # Delete input files after 3 days in dev
  s3_output_expiration_days = 7   # Delete output files after 7 days in dev

  # Inference configuration (matches config/inference_config.yaml)
  inference_config = {
    fmax_hz              = 2000
    n_fft                = 1024
    hop_length           = 256
    width_pixels         = 256
    height_pixels        = 256
    window_duration      = 1.0
    stride_duration      = 0.4
    batch_size           = 32
    confidence_threshold = 0.5
    target_class         = "anemonefish"
    min_event_duration   = 0.2
    min_gap_duration     = 0.1
    smoothing_window     = 5
    model_classes        = "noise,anemonefish,biological"
  }

  tags = var.tags

  depends_on = [module.ecr]
}
