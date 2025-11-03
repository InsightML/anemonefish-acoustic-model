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
    bucket         = null  # Set your Terraform state bucket
    key            = "anemonefish-inference/prod/terraform.tfstate"
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
  environment  = "prod"

  image_tag_mutability    = "IMMUTABLE"  # Immutable for production
  scan_on_push            = true
  lifecycle_policy_enabled = true
  max_image_count          = 20  # Keep more images in prod

  tags = var.tags
}

# ============================================================================
# INFERENCE INFRASTRUCTURE
# ============================================================================

module "inference" {
  source = "../../modules/inference"

  project_name = var.project_name
  environment  = "prod"
  aws_region   = data.aws_region.current.name

  # Model configuration
  model_s3_bucket = var.model_s3_bucket
  model_s3_key    = var.model_s3_key
  model_version   = var.model_version

  # Lambda configuration
  lambda_timeout                   = 900  # 15 minutes
  lambda_memory_size              = 3008  # Max memory for large audio processing
  lambda_reserved_concurrent_executions = var.lambda_reserved_concurrent_executions

  # Lambda container image (must specify exact tag/version in prod)
  lambda_image_uri = var.lambda_image_uri

  # API Gateway configuration
  api_gateway_stage_name = "api"
  cors_allowed_origins   = var.cors_allowed_origins  # Restrict to frontend domain
  enable_api_key         = var.enable_api_key

  # Logging
  log_retention_days = 30  # Longer retention for prod

  # S3 lifecycle
  s3_lifecycle_enabled      = true
  s3_input_expiration_days  = 7   # Delete input files after 7 days
  s3_output_expiration_days = 30  # Keep output files for 30 days

  # Inference configuration
  inference_config = var.inference_config

  tags = var.tags

  depends_on = [module.ecr]
}
