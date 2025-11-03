output "repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.inference.repository_url
}

output "repository_arn" {
  description = "ARN of the ECR repository"
  value       = aws_ecr_repository.inference.arn
}

output "repository_name" {
  description = "Name of the ECR repository"
  value       = aws_ecr_repository.inference.name
}

output "registry_id" {
  description = "Registry ID of the ECR repository"
  value       = aws_ecr_repository.inference.registry_id
}
