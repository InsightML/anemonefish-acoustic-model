variable "project_name" {
  description = "Project name prefix for resource naming"
  type        = string
  default     = "anemonefish-inference"
}

variable "model_s3_bucket" {
  description = "S3 bucket name for model artifacts (must exist or be created separately)"
  type        = string
}

variable "model_s3_key" {
  description = "S3 key path to the model file"
  type        = string
  default     = "models/v1.0/best_model.keras"
}

variable "model_version" {
  description = "Model version identifier"
  type        = string
  default     = "v1.0"
}

variable "lambda_image_uri" {
  description = "ECR image URI for Lambda container (required in prod - use specific tag/version)"
  type        = string
}

variable "lambda_reserved_concurrent_executions" {
  description = "Reserved concurrent executions for Lambda (null = unlimited)"
  type        = number
  default     = null
}

variable "cors_allowed_origins" {
  description = "List of allowed CORS origins for production (restrict to frontend domain)"
  type        = list(string)
  default     = []  # Must be set explicitly in prod
}

variable "enable_api_key" {
  description = "Enable API key authentication"
  type        = bool
  default     = true  # Default to enabled in prod
}

variable "inference_config" {
  description = "Inference configuration parameters"
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

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}
