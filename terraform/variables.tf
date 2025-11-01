variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for resource naming and tagging"
  type        = string
  default     = "anemonefish-training"
}

variable "instance_type" {
  description = "EC2 instance type (GPU instance)"
  type        = string
  default     = "g4dn.xlarge"
}

variable "ebs_volume_size" {
  description = "Size of EBS volume in GB"
  type        = number
  default     = 100
}

variable "ssh_key_name" {
  description = "Name of existing AWS key pair for SSH access"
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into the instance"
  type        = string
  default     = "0.0.0.0/0"
}

variable "use_spot_instance" {
  description = "Whether to use spot instances for cost savings (60-70% cheaper)"
  type        = bool
  default     = false
}

variable "spot_max_price" {
  description = "Maximum price for spot instances (leave empty for on-demand price)"
  type        = string
  default     = ""
}

variable "s3_bucket_name" {
  description = "S3 bucket name for training data and models (must be globally unique)"
  type        = string
  default     = ""
}

variable "auto_terminate" {
  description = "Whether to auto-terminate the instance after training completes"
  type        = bool
  default     = true
}

variable "enable_tensorboard" {
  description = "Whether to enable TensorBoard web interface"
  type        = bool
  default     = true
}
