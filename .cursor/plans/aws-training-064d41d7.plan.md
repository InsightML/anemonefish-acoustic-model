<!-- 064d41d7-8e15-4efc-8072-f84064bf6d66 bacf5e72-0f51-414a-8c91-1344af3cfc19 -->
# AWS Training Deployment with Terraform

## Architecture Overview

Deploy the `target_to_noise_classifier.py` training script to AWS using:

- **EC2 GPU instance** (configurable type) for training
- **S3 bucket** for data storage and model artifacts
- **TensorBoard** web interface for training monitoring
- **Auto-termination** after training completes to minimize costs

## Infrastructure Components

### 1. Terraform Configuration Structure

```
terraform/
├── main.tf                 # Main infrastructure definition
├── variables.tf            # Input variables (GPU type, storage, region, etc.)
├── outputs.tf              # Outputs (instance IP, S3 bucket, connection info)
├── user_data.sh            # EC2 initialization script
├── training_runner.sh      # Training execution script (runs on EC2)
└── terraform.tfvars        # Variable values (gitignored)
```

**Key Resources in `main.tf`:**

- EC2 instance with Deep Learning AMI (Ubuntu, CUDA pre-installed)
- Security group allowing SSH (port 22) and TensorBoard (port 6006)
- S3 bucket for training data and model outputs
- IAM role/instance profile with S3 read/write permissions
- EBS volume for local storage

**Variables in `variables.tf`:**

- `instance_type` (default: "g4dn.xlarge")
- `ebs_volume_size` (default: 100 GB)
- `aws_region` (default: "us-east-1")
- `project_name` (for resource naming/tagging)
- `ssh_key_name` (existing AWS key pair)
- `allowed_ssh_cidr` (IP whitelist for SSH access)

### 2. Data Upload Script

**`scripts/aws_deployment/upload_data_to_s3.py`:**

- Upload training dataset from local path to S3
- Source: `/data/2_training_datasets/v2_biological/`
- Also upload: preprocessing config YAML, requirements.txt, src/ package code
- Use boto3 with progress bars (tqdm)

### 3. EC2 Initialization (`user_data.sh`)

Auto-runs on instance launch:

1. Install system dependencies (if needed beyond Deep Learning AMI)
2. Setup conda environment `anemonefish_model`
3. Install TensorFlow for GPU (standard Linux version, not macOS-specific)
4. Sync training data from S3 to `/home/ubuntu/Clown_Fish_Acoustics/data/`
5. Install anemonefish_acoustics package (`pip install -e .`)
6. Create systemd service or use tmux to run training script
7. Launch TensorBoard pointing to logs directory
8. Execute `training_runner.sh`

### 4. Training Execution Script (`training_runner.sh`)

Runs on EC2 after initialization:

1. Activate conda environment
2. Run `python scripts/training/target_to_noise_classifier.py`
3. Monitor training completion
4. Sync results to S3:

   - Models directory → `s3://bucket/models/`
   - Logs directory → `s3://bucket/logs/`
   - Training history → `s3://bucket/results/`

5. Trigger instance termination (via AWS CLI or API call)

### 5. Config File Modifications

**Update `preprocessing_config_v2_biological.yaml` dynamically:**

- Replace `workspace_base_path` from NAS path to `/home/ubuntu/Clown_Fish_Acoustics`
- Use `sed` or Python script during initialization

**Create `requirements-aws.txt` for Linux GPU:**

Based on your pip list, create AWS-compatible requirements:

- Replace `tensorflow-macos==2.16.2` + `tensorflow-metal==1.2.0` with `tensorflow[and-cuda]==2.18.0`
- Include core packages: `keras==3.11.3`, `keras-tuner==1.4.7`, `boto3==1.38.25`
- Include ML/audio stack: `numpy==1.26.4`, `pandas==2.2.3`, `scikit-learn==1.6.1`, `librosa==0.10.2.post1`, `soundfile==0.13.1`
- Include visualization: `matplotlib==3.10.1`, `seaborn==0.13.2`, `plotly==6.1.2`
- Include other essentials: `PyYAML==6.0.2`, `tqdm==4.67.1`, `pillow==11.1.0`
- All other packages from your environment that are needed for training

### 6. Helper Scripts

**`scripts/aws_deployment/deploy.sh`:**

- Wrapper script to:

  1. Upload data to S3 (call upload_data_to_s3.py)
  2. Run `terraform apply`
  3. Wait for instance to be ready
  4. Display connection instructions

**`scripts/aws_deployment/connect.sh`:**

- SSH into the instance with port forwarding for TensorBoard
- Command: `ssh -i ~/.ssh/key.pem -L 6006:localhost:6006 ubuntu@<instance-ip>`

**`scripts/aws_deployment/download_results.sh`:**

- Download trained models and logs from S3 to local machine
- Destination: `models/` and `logs/` directories

**`scripts/aws_deployment/cleanup.sh`:**

- Run `terraform destroy` to remove all AWS resources

## Implementation Files

### Key Files to Create:

1. `terraform/main.tf` - Complete infrastructure definition
2. `terraform/variables.tf` - All configurable parameters
3. `terraform/outputs.tf` - Instance IP, S3 bucket name, connection commands
4. `terraform/user_data.sh` - EC2 bootstrap script
5. `terraform/training_runner.sh` - Training orchestration script
6. `scripts/aws_deployment/upload_data_to_s3.py` - Data upload script
7. `scripts/aws_deployment/deploy.sh` - Deployment orchestration
8. `scripts/aws_deployment/connect.sh` - SSH helper
9. `scripts/aws_deployment/download_results.sh` - Results retrieval
10. `terraform/terraform.tfvars.example` - Example configuration
11. `terraform/.gitignore` - Ignore sensitive files
12. `terraform/README.md` - Deployment documentation

### Key Files to Modify:

1. `requirements.txt` - Add conditional Linux vs macOS dependencies (or create `requirements-aws.txt`)
2. `.gitignore` - Add terraform state files, tfvars

## Auto-Termination Strategy

Two options:

- **Option A (Recommended):** Training script calls AWS CLI/SDK to terminate instance on completion
- **Option B:** CloudWatch alarm triggers termination after CPU/GPU idle for X minutes

Implementation in `training_runner.sh`:

```bash
# After training completes
aws s3 sync /home/ubuntu/Clown_Fish_Acoustics/models/ s3://bucket/models/
aws s3 sync /home/ubuntu/Clown_Fish_Acoustics/logs/ s3://bucket/logs/
# Self-terminate
INSTANCE_ID=$(ec2-metadata --instance-id | cut -d " " -f 2)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
```

## Security Considerations

- SSH access restricted to specific CIDR (your IP)
- TensorBoard port also restricted to your IP
- S3 bucket with server-side encryption enabled
- IAM role follows least-privilege (only S3 access)
- Terraform state stored locally (consider S3 backend for team usage)

## Cost Optimization

- Auto-termination prevents runaway costs
- Spot instances option (add variable, 60-70% cost savings)
- S3 lifecycle policy to archive old training runs to Glacier
- CloudWatch billing alarm (configurable threshold)

## Workflow

1. **Initial Setup:** Configure `terraform.tfvars` with AWS credentials, key pair, IP whitelist
2. **Deploy:** Run `./scripts/aws_deployment/deploy.sh`
3. **Monitor:** Access TensorBoard at `http://localhost:6006` (via SSH tunnel)
4. **Check Status:** SSH into instance to monitor progress
5. **Retrieve Results:** After auto-termination, run `./scripts/aws_deployment/download_results.sh`
6. **Cleanup (if needed):** Run `./scripts/aws_deployment/cleanup.sh` to destroy infrastructure

## Future Enhancements

- Add support for multi-GPU training (distributed strategy)
- Create AMI snapshot with pre-installed dependencies for faster startup
- Add CloudWatch dashboard for GPU utilization monitoring
- Support for Weights & Biases or MLflow integration
- Parameterized training (pass hyperparameters via Terraform variables)

### To-dos

- [ ] Create terraform directory structure with main.tf, variables.tf, outputs.tf, and template files
- [ ] Implement Terraform resources: EC2 instance, security groups, S3 bucket, IAM roles, and EBS volume
- [ ] Write user_data.sh script to bootstrap EC2 instance with conda, dependencies, and data sync from S3
- [ ] Write training_runner.sh to execute training, sync results to S3, and trigger auto-termination
- [ ] Write Python script to upload training data, config, and source code to S3 bucket
- [ ] Create deploy.sh, connect.sh, download_results.sh, and cleanup.sh helper scripts
- [ ] Create requirements-aws.txt with Linux-compatible TensorFlow and add missing dependencies like keras-tuner and boto3
- [ ] Write terraform/README.md with setup instructions, usage guide, and troubleshooting tips
- [ ] Add logic to dynamically update workspace_base_path in YAML config for AWS environment
- [ ] Update .gitignore to exclude Terraform state files, tfvars, and AWS credentials