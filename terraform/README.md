# AWS Training Deployment with Terraform

This directory contains Terraform infrastructure-as-code to deploy the Anemonefish acoustic classifier training pipeline to AWS. The infrastructure automatically provisions GPU instances, manages data storage, and handles training execution with auto-termination to minimize costs.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Monitoring](#monitoring)
- [Cost Optimization](#cost-optimization)
- [Troubleshooting](#troubleshooting)
- [Infrastructure Details](#infrastructure-details)

## Architecture Overview

### Components

```
???????????????????????????????????????????????????????????????
?                      AWS Cloud                              ?
?                                                             ?
?  ??????????????????         ???????????????????           ?
?  ?  EC2 Instance  ???????????   S3 Bucket     ?           ?
?  ?  (GPU)         ?         ?  (Training Data ?           ?
?  ?  - Training    ?         ?   & Models)     ?           ?
?  ?  - TensorBoard ?         ???????????????????           ?
?  ??????????????????                                        ?
?         ?                                                   ?
?         ? Auto-terminate                                    ?
?         ?                                                   ?
?  ??????????????????                                        ?
?  ?  IAM Role      ?                                        ?
?  ?  - S3 Access   ?                                        ?
?  ?  - Termination ?                                        ?
?  ??????????????????                                        ?
?                                                             ?
???????????????????????????????????????????????????????????????
         ?                            ?
         ? SSH (port 22)              ? Results
         ? TensorBoard (port 6006)    ? Download
         ?                            ?
    ????????????               ????????????
    ?   You    ?               ?   You    ?
    ????????????               ????????????
```

### Resources Created

- **EC2 Instance**: GPU-enabled Deep Learning AMI for training
- **S3 Bucket**: Stores training data, models, and logs
- **Security Group**: Restricts access to SSH and TensorBoard
- **IAM Role**: Grants instance permissions for S3 and self-termination
- **EBS Volume**: Provides local storage for training
- **CloudWatch Alarm**: Monitors estimated charges

## Prerequisites

### 1. Required Tools

Install the following tools on your local machine:

```bash
# Terraform
brew install terraform  # macOS
# or download from https://www.terraform.io/downloads

# AWS CLI
brew install awscli     # macOS
# or download from https://aws.amazon.com/cli/

# Python 3 with boto3
pip install boto3 tqdm
```

### 2. AWS Account Setup

1. **Create AWS Account**: If you don't have one, sign up at [aws.amazon.com](https://aws.amazon.com)

2. **Configure AWS Credentials**:
   ```bash
   aws configure
   ```
   Provide:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region (e.g., `us-east-1`)
   - Output format (e.g., `json`)

3. **Create EC2 Key Pair**:
   ```bash
   aws ec2 create-key-pair \
     --key-name anemonefish-training \
     --query 'KeyMaterial' \
     --output text > ~/.ssh/anemonefish-training.pem
   
   chmod 400 ~/.ssh/anemonefish-training.pem
   ```

4. **Get Your Public IP**:
   ```bash
   curl ifconfig.me
   ```
   You'll need this for SSH access configuration.

### 3. Training Data

Ensure your training data is available at:
```
/data/2_training_datasets/v2_biological/
```

Or specify a different path when uploading to S3.

## Quick Start

### 1. Configure Terraform Variables

```bash
cd terraform/
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
# Required - AWS Configuration
aws_region       = "us-east-1"
ssh_key_name     = "anemonefish-training"
allowed_ssh_cidr = "YOUR.IP.ADDRESS/32"  # Replace with your IP

# Optional - Resource Configuration
project_name     = "anemonefish-training"
instance_type    = "g4dn.xlarge"
ebs_volume_size  = 100

# Optional - Cost Optimization
use_spot_instance = false

# Optional - Features
auto_terminate      = true
enable_tensorboard  = true
```

### 2. Deploy Infrastructure

Use the deployment script (recommended):

```bash
cd ..
./scripts/aws_deployment/deploy.sh
```

This will:
1. Upload training data to S3
2. Initialize Terraform
3. Create AWS infrastructure
4. Start training automatically

Or deploy manually:

```bash
# Upload data to S3 first
python scripts/aws_deployment/upload_data_to_s3.py \
  --bucket your-bucket-name \
  --data-dir /path/to/training/data

# Deploy with Terraform
cd terraform/
terraform init
terraform plan
terraform apply
```

### 3. Monitor Training

**Option A: SSH with TensorBoard** (recommended)
```bash
./scripts/aws_deployment/connect.sh --tensorboard
```
Then open http://localhost:6006 in your browser.

**Option B: Check Training Logs**
```bash
./scripts/aws_deployment/connect.sh --logs
```

**Option C: Check Status**
```bash
./scripts/aws_deployment/connect.sh --status
```

### 4. Download Results

After training completes:

```bash
./scripts/aws_deployment/download_results.sh
```

Results will be saved to:
- `models/` - Trained model files
- `logs/` - TensorBoard logs
- `training.log` - Training execution log

### 5. Cleanup

**Important**: Destroy infrastructure to stop incurring costs:

```bash
./scripts/aws_deployment/cleanup.sh
```

## Configuration

### Instance Types

Choose based on your performance and budget needs:

| Instance Type | GPU          | vCPUs | RAM   | On-Demand | Spot (est.) |
|--------------|--------------|-------|-------|-----------|-------------|
| g4dn.xlarge  | 1x T4        | 4     | 16GB  | $0.526/hr | $0.158/hr   |
| g4dn.2xlarge | 1x T4        | 8     | 32GB  | $0.752/hr | $0.226/hr   |
| g5.xlarge    | 1x A10G      | 4     | 16GB  | $1.006/hr | $0.302/hr   |
| g5.2xlarge   | 1x A10G      | 8     | 32GB  | $1.212/hr | $0.364/hr   |
| p3.2xlarge   | 1x V100      | 8     | 61GB  | $3.06/hr  | $0.918/hr   |

**Recommendation**: Start with `g4dn.xlarge` for cost-effective training.

### Spot Instances

Save 60-70% by using spot instances:

```hcl
use_spot_instance = true
spot_max_price    = ""  # Empty = on-demand price as max
```

**Caution**: Spot instances can be interrupted with 2-minute notice.

### Auto-Termination

Instance automatically terminates after training completes:

```hcl
auto_terminate = true  # Default: enabled
```

**Disable for**:
- Debugging
- Multiple training runs
- Interactive experimentation

## Usage

### Helper Scripts

All scripts are located in `scripts/aws_deployment/`:

#### 1. `deploy.sh`
Deploy the complete infrastructure:
```bash
./scripts/aws_deployment/deploy.sh [OPTIONS]

Options:
  --skip-upload      Skip uploading data to S3
  --skip-terraform   Skip Terraform apply
```

#### 2. `connect.sh`
Connect to the training instance:
```bash
./scripts/aws_deployment/connect.sh [OPTIONS]

Options:
  --tensorboard    Connect with TensorBoard port forwarding
  --logs           Show training logs (tail -f)
  --status         Check training and system status
  --command CMD    Execute a command on the instance
```

Examples:
```bash
# Interactive shell
./scripts/aws_deployment/connect.sh

# TensorBoard
./scripts/aws_deployment/connect.sh --tensorboard

# Monitor logs
./scripts/aws_deployment/connect.sh --logs

# Check GPU usage
./scripts/aws_deployment/connect.sh --command "nvidia-smi"
```

#### 3. `download_results.sh`
Download training results from S3:
```bash
./scripts/aws_deployment/download_results.sh [OPTIONS]

Options:
  --output-dir DIR    Output directory (default: workspace root)
  --models-only       Download only model files
  --logs-only         Download only log files
```

#### 4. `cleanup.sh`
Destroy all infrastructure:
```bash
./scripts/aws_deployment/cleanup.sh [OPTIONS]

Options:
  --skip-backup       Skip downloading results before cleanup
  --delete-s3-data    Delete all S3 data (WARNING: irreversible!)
```

### Manual Operations

#### Check Instance Status
```bash
cd terraform/
terraform output connection_instructions
```

#### SSH into Instance
```bash
ssh -i ~/.ssh/anemonefish-training.pem ubuntu@<instance-ip>
```

#### Monitor Training
```bash
# View logs
tail -f /home/ubuntu/training.log

# Check service status
systemctl status training.service

# Check GPU usage
nvidia-smi
watch -n 1 nvidia-smi
```

#### Manual Data Upload
```bash
python scripts/aws_deployment/upload_data_to_s3.py \
  --bucket your-bucket-name \
  --data-dir /path/to/data \
  --region us-east-1
```

## Monitoring

### TensorBoard

Access TensorBoard through SSH tunnel:

1. Connect with port forwarding:
   ```bash
   ./scripts/aws_deployment/connect.sh --tensorboard
   ```

2. Open browser to: http://localhost:6006

### Training Logs

View real-time logs:
```bash
./scripts/aws_deployment/connect.sh --logs
```

### AWS Console

Monitor resources in AWS Console:
- **EC2**: Instance status, CPU/GPU metrics
- **S3**: Data and results storage
- **CloudWatch**: Logs and metrics
- **Billing**: Cost tracking

## Cost Optimization

### 1. Enable Auto-Termination
Ensure `auto_terminate = true` to stop instances after training.

### 2. Use Spot Instances
Set `use_spot_instance = true` for 60-70% savings.

### 3. Right-Size Instance
Start with `g4dn.xlarge` and scale up only if needed.

### 4. Monitor Spending
Set up billing alerts:
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name training-budget \
  --alarm-description "Alert when charges exceed $50" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 21600 \
  --evaluation-periods 1 \
  --threshold 50 \
  --comparison-operator GreaterThanThreshold
```

### 5. Clean Up Promptly
Always run cleanup after downloading results:
```bash
./scripts/aws_deployment/cleanup.sh
```

### Estimated Costs

For a typical training run (4-8 hours):

| Instance       | Duration | Cost (On-Demand) | Cost (Spot) |
|----------------|----------|------------------|-------------|
| g4dn.xlarge    | 4h       | $2.10            | $0.63       |
| g4dn.xlarge    | 8h       | $4.21            | $1.26       |
| g5.xlarge      | 4h       | $4.02            | $1.21       |
| g5.xlarge      | 8h       | $8.05            | $2.42       |

Plus S3 storage: ~$0.023/GB/month

## Troubleshooting

### Common Issues

#### 1. Key Pair Not Found
**Error**: `The key pair 'xxx' does not exist`

**Solution**:
```bash
# Create key pair
aws ec2 create-key-pair \
  --key-name your-key-name \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/your-key-name.pem

chmod 400 ~/.ssh/your-key-name.pem
```

#### 2. Permission Denied (SSH)
**Error**: `Permission denied (publickey)`

**Solution**:
```bash
# Fix key permissions
chmod 400 ~/.ssh/your-key-name.pem

# Use correct username
ssh -i ~/.ssh/your-key-name.pem ubuntu@<ip>  # Not 'ec2-user'
```

#### 3. Instance Limit Exceeded
**Error**: `You have requested more instances than your current instance limit`

**Solution**:
Request limit increase in AWS Console:
1. Go to EC2 ? Limits
2. Select "Running On-Demand GPU instances"
3. Click "Request limit increase"

#### 4. S3 Bucket Already Exists
**Error**: `BucketAlreadyExists`

**Solution**:
Set a unique bucket name in `terraform.tfvars`:
```hcl
s3_bucket_name = "anemonefish-training-unique-12345"
```

#### 5. Training Failed
**Problem**: Training script exits with error

**Diagnosis**:
```bash
# Check logs
./scripts/aws_deployment/connect.sh --logs

# Or SSH and investigate
./scripts/aws_deployment/connect.sh
tail -100 /home/ubuntu/training.log
tail -100 /home/ubuntu/init.log
```

**Common causes**:
- Missing training data
- Incorrect config file path
- CUDA/GPU issues
- Out of memory

#### 6. TensorBoard Not Accessible
**Problem**: Can't access http://localhost:6006

**Solution**:
```bash
# Reconnect with proper port forwarding
./scripts/aws_deployment/connect.sh --tensorboard

# Check TensorBoard service
./scripts/aws_deployment/connect.sh --command "systemctl status tensorboard"
```

#### 7. Instance Doesn't Auto-Terminate
**Problem**: Instance still running after training

**Causes**:
- `auto_terminate = false` in config
- Training script failed before termination
- IAM permissions insufficient

**Solution**:
```bash
# Manually terminate
cd terraform/
terraform destroy

# Or terminate specific instance
aws ec2 terminate-instances --instance-ids i-xxxxx
```

### Debug Mode

Enable verbose logging:

```bash
# SSH into instance
./scripts/aws_deployment/connect.sh

# Check initialization log
cat /home/ubuntu/init.log

# Check training log
cat /home/ubuntu/training.log

# Check systemd logs
journalctl -u training.service
journalctl -u tensorboard.service

# Check disk space
df -h

# Check GPU
nvidia-smi
```

### Getting Help

1. Check logs: `/home/ubuntu/init.log` and `/home/ubuntu/training.log`
2. Verify AWS resources in Console
3. Check Terraform state: `terraform show`
4. Review security group rules
5. Verify IAM permissions

## Infrastructure Details

### Terraform Resources

| Resource | Purpose |
|----------|---------|
| `aws_instance.training_instance` | EC2 GPU instance for training |
| `aws_s3_bucket.training_bucket` | Storage for data and models |
| `aws_security_group.training_sg` | Network access control |
| `aws_iam_role.training_instance_role` | Instance permissions |
| `aws_iam_instance_profile` | Attach IAM role to instance |

### Security

- **SSH Access**: Restricted to your IP address
- **TensorBoard**: Restricted to your IP address
- **S3 Encryption**: AES-256 server-side encryption
- **IAM**: Least-privilege access (S3 only)
- **Key Pairs**: Required for SSH access

### Data Flow

1. **Upload Phase**:
   - Local data ? S3 bucket (via `upload_data_to_s3.py`)

2. **Initialization Phase**:
   - EC2 launches with user_data script
   - Downloads code and data from S3
   - Installs dependencies
   - Creates conda environment

3. **Training Phase**:
   - Runs training script
   - Logs to TensorBoard
   - Saves checkpoints

4. **Completion Phase**:
   - Uploads results to S3
   - Auto-terminates instance (if enabled)

5. **Download Phase**:
   - Retrieves results from S3 to local machine

## Advanced Configuration

### Custom Training Script

To use a different training script, modify `user_data.sh`:

```bash
TRAINING_SCRIPT="scripts/training/your_script.py"
```

### Multi-GPU Training

For instances with multiple GPUs:

1. Choose multi-GPU instance (e.g., `g5.12xlarge`)
2. Modify training script to use distributed strategy:
   ```python
   strategy = tf.distribute.MirroredStrategy()
   with strategy.scope():
       model = build_model()
   ```

### Custom AMI

For faster startup, create a custom AMI:

1. Launch instance and install dependencies
2. Create AMI snapshot
3. Use AMI ID in `main.tf`:
   ```hcl
   ami = "ami-xxxxx"  # Your custom AMI
   ```

## Next Steps

- [ ] Configure `terraform.tfvars`
- [ ] Upload training data to S3
- [ ] Deploy infrastructure
- [ ] Monitor training progress
- [ ] Download results
- [ ] Clean up resources

## License

This infrastructure code is part of the Anemonefish Acoustics project.

## Support

For issues or questions:
1. Check this README
2. Review Terraform documentation
3. Check AWS documentation
4. Review training script logs
