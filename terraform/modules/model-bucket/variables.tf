variable "project_name" {
  description = "Project name prefix for resource naming"
  type        = string
  default     = "anemonefish-inference"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "bucket_name" {
  description = "S3 bucket name for model artifacts. If null, auto-generated from project_name and environment"
  type        = string
  default     = null
}

variable "lifecycle_policy_enabled" {
  description = "Enable lifecycle policy for transitioning to Glacier"
  type        = bool
  default     = true
}

variable "glacier_transition_days" {
  description = "Days before transitioning to Glacier storage class"
  type        = number
  default     = 90
}

variable "delete_old_versions" {
  description = "Enable deletion of old non-current versions"
  type        = bool
  default     = false
}

variable "noncurrent_version_expiration_days" {
  description = "Days before non-current versions are deleted"
  type        = number
  default     = 365
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}
