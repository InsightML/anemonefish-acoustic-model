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
  description = "ECR image URI for Lambda container. If null, uses ECR module output with :latest tag"
  type        = string
  default     = null
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}
