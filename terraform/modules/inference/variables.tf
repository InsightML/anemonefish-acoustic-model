variable "project_name" {
  description = "Project name prefix for resource naming"
  type        = string
  default     = "anemonefish-inference"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "model_s3_bucket" {
  description = "S3 bucket name for model artifacts (must already exist or be created separately)"
  type        = string
}

variable "model_s3_key" {
  description = "S3 key path to the model file (e.g., models/v1.0/best_model.keras)"
  type        = string
  default     = "models/v1.0/best_model.keras"
}

variable "model_version" {
  description = "Model version identifier"
  type        = string
  default     = "v1.0"
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds (max 900 for 15 minutes)"
  type        = number
  default     = 900
}

variable "lambda_memory_size" {
  description = "Lambda function memory size in MB (recommended: 3008 for large audio processing)"
  type        = number
  default     = 3008
}

variable "lambda_reserved_concurrent_executions" {
  description = "Reserved concurrent executions for Lambda (null = unlimited)"
  type        = number
  default     = null
}

variable "lambda_image_uri" {
  description = "ECR image URI for Lambda container (e.g., 123456789.dkr.ecr.us-east-1.amazonaws.com/anemonefish-inference:latest)"
  type        = string
}

variable "api_gateway_stage_name" {
  description = "API Gateway stage name"
  type        = string
  default     = "api"
}

variable "cors_allowed_origins" {
  description = "List of allowed CORS origins (use ['*'] for all origins in dev)"
  type        = list(string)
  default     = ["*"]
}

variable "enable_api_key" {
  description = "Enable API key authentication for API Gateway"
  type        = bool
  default     = false
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "s3_lifecycle_enabled" {
  description = "Enable S3 lifecycle policies for automatic cleanup"
  type        = bool
  default     = true
}

variable "s3_input_expiration_days" {
  description = "Days until input bucket objects are deleted (0 = disable)"
  type        = number
  default     = 7
}

variable "s3_output_expiration_days" {
  description = "Days until output bucket objects are deleted (0 = disable)"
  type        = number
  default     = 30
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

variable "inference_config" {
  description = "Inference configuration parameters (will be passed as environment variables)"
  type = object({
    fmax_hz                = optional(number, 2000)
    n_fft                  = optional(number, 1024)
    hop_length             = optional(number, 256)
    width_pixels           = optional(number, 256)
    height_pixels          = optional(number, 256)
    window_duration        = optional(number, 1.0)
    stride_duration        = optional(number, 0.4)
    batch_size             = optional(number, 32)
    confidence_threshold   = optional(number, 0.5)
    target_class           = optional(string, "anemonefish")
    min_event_duration     = optional(number, 0.2)
    min_gap_duration       = optional(number, 0.1)
    smoothing_window       = optional(number, 5)
    model_classes          = optional(string, "noise,anemonefish,biological")
  })
  default = {}
}
