output "instance_id" {
  description = "ID of the EC2 training instance"
  value       = aws_instance.training_instance.id
}

output "instance_public_ip" {
  description = "Public IP address of the EC2 training instance"
  value       = aws_instance.training_instance.public_ip
}

output "instance_public_dns" {
  description = "Public DNS name of the EC2 training instance"
  value       = aws_instance.training_instance.public_dns
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for training data and models"
  value       = aws_s3_bucket.training_bucket.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.training_bucket.arn
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/${var.ssh_key_name}.pem ubuntu@${aws_instance.training_instance.public_ip}"
}

output "ssh_command_with_tensorboard" {
  description = "SSH command with TensorBoard port forwarding"
  value       = "ssh -i ~/.ssh/${var.ssh_key_name}.pem -L 6006:localhost:6006 ubuntu@${aws_instance.training_instance.public_ip}"
}

output "tensorboard_url" {
  description = "TensorBoard URL (accessible via SSH tunnel)"
  value       = var.enable_tensorboard ? "http://localhost:6006" : "TensorBoard disabled"
}

output "connection_instructions" {
  description = "Instructions for connecting to the instance"
  value       = <<-EOT
    === Connection Instructions ===
    
    1. SSH into the instance:
       ${format("ssh -i ~/.ssh/%s.pem ubuntu@%s", var.ssh_key_name, aws_instance.training_instance.public_ip)}
    
    2. SSH with TensorBoard port forwarding:
       ${format("ssh -i ~/.ssh/%s.pem -L 6006:localhost:6006 ubuntu@%s", var.ssh_key_name, aws_instance.training_instance.public_ip)}
       Then open http://localhost:6006 in your browser
    
    3. Monitor training logs:
       tail -f /home/ubuntu/training.log
    
    4. Check instance status:
       systemctl status training.service
    
    5. Download results from S3:
       aws s3 sync s3://${aws_s3_bucket.training_bucket.id}/models/ ./models/
       aws s3 sync s3://${aws_s3_bucket.training_bucket.id}/logs/ ./logs/
  EOT
}

output "aws_region" {
  description = "AWS region where resources are deployed"
  value       = var.aws_region
}

output "instance_type" {
  description = "EC2 instance type used"
  value       = var.instance_type
}

output "ami_id" {
  description = "AMI ID used for the instance"
  value       = data.aws_ami.deep_learning.id
}
