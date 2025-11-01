terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data source to get the latest Deep Learning AMI
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch * (Ubuntu 20.04)*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# S3 bucket for training data and model outputs
resource "aws_s3_bucket" "training_bucket" {
  bucket = var.s3_bucket_name != "" ? var.s3_bucket_name : "${var.project_name}-${random_id.bucket_suffix.hex}"

  tags = {
    Name    = "${var.project_name}-training-bucket"
    Project = var.project_name
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Enable versioning for the S3 bucket
resource "aws_s3_bucket_versioning" "training_bucket_versioning" {
  bucket = aws_s3_bucket.training_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Enable server-side encryption for the S3 bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "training_bucket_encryption" {
  bucket = aws_s3_bucket.training_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 bucket lifecycle policy to archive old training runs
resource "aws_s3_bucket_lifecycle_configuration" "training_bucket_lifecycle" {
  bucket = aws_s3_bucket.training_bucket.id

  rule {
    id     = "archive-old-runs"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

# IAM role for EC2 instance
resource "aws_iam_role" "training_instance_role" {
  name = "${var.project_name}-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name    = "${var.project_name}-instance-role"
    Project = var.project_name
  }
}

# IAM policy for S3 access
resource "aws_iam_role_policy" "training_instance_s3_policy" {
  name = "${var.project_name}-s3-policy"
  role = aws_iam_role.training_instance_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${aws_s3_bucket.training_bucket.arn}",
          "${aws_s3_bucket.training_bucket.arn}/*"
        ]
      }
    ]
  })
}

# IAM policy for EC2 termination (for auto-termination)
resource "aws_iam_role_policy" "training_instance_termination_policy" {
  count = var.auto_terminate ? 1 : 0
  name  = "${var.project_name}-termination-policy"
  role  = aws_iam_role.training_instance_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:TerminateInstances",
          "ec2:DescribeInstances"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "ec2:ResourceTag/Project" = var.project_name
          }
        }
      }
    ]
  })
}

# IAM instance profile
resource "aws_iam_instance_profile" "training_instance_profile" {
  name = "${var.project_name}-instance-profile"
  role = aws_iam_role.training_instance_role.name

  tags = {
    Name    = "${var.project_name}-instance-profile"
    Project = var.project_name
  }
}

# Security group for the EC2 instance
resource "aws_security_group" "training_sg" {
  name        = "${var.project_name}-security-group"
  description = "Security group for training instance"

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    description = "SSH access"
  }

  # TensorBoard access (if enabled)
  dynamic "ingress" {
    for_each = var.enable_tensorboard ? [1] : []
    content {
      from_port   = 6006
      to_port     = 6006
      protocol    = "tcp"
      cidr_blocks = [var.allowed_ssh_cidr]
      description = "TensorBoard access"
    }
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  tags = {
    Name    = "${var.project_name}-security-group"
    Project = var.project_name
  }
}

# User data script
data "template_file" "user_data" {
  template = file("${path.module}/user_data.sh")

  vars = {
    s3_bucket_name   = aws_s3_bucket.training_bucket.id
    project_name     = var.project_name
    aws_region       = var.aws_region
    auto_terminate   = var.auto_terminate
    enable_tensorboard = var.enable_tensorboard
  }
}

# EC2 instance for training
resource "aws_instance" "training_instance" {
  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type
  key_name               = var.ssh_key_name
  iam_instance_profile   = aws_iam_instance_profile.training_instance_profile.name
  vpc_security_group_ids = [aws_security_group.training_sg.id]

  # User data for instance initialization
  user_data = data.template_file.user_data.rendered

  # Root volume configuration
  root_block_device {
    volume_size           = var.ebs_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true

    tags = {
      Name    = "${var.project_name}-root-volume"
      Project = var.project_name
    }
  }

  # Spot instance configuration (if enabled)
  dynamic "instance_market_options" {
    for_each = var.use_spot_instance ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        max_price                      = var.spot_max_price != "" ? var.spot_max_price : null
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  tags = {
    Name    = "${var.project_name}-training-instance"
    Project = var.project_name
  }

  # Ensure instance doesn't get replaced unnecessarily
  lifecycle {
    ignore_changes = [ami, user_data]
  }
}

# CloudWatch billing alarm (optional but recommended)
resource "aws_cloudwatch_metric_alarm" "billing_alarm" {
  alarm_name          = "${var.project_name}-billing-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "21600" # 6 hours
  statistic           = "Maximum"
  threshold           = "100" # $100 threshold
  alarm_description   = "This metric monitors estimated AWS charges"
  alarm_actions       = [] # Add SNS topic ARN here if you want email notifications

  dimensions = {
    Currency = "USD"
  }

  tags = {
    Name    = "${var.project_name}-billing-alarm"
    Project = var.project_name
  }
}
