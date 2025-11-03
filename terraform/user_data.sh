#!/bin/bash
set -e

# Logging setup
LOGFILE="/home/ubuntu/init.log"
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "Starting EC2 Instance Initialization"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Instance ID: $(ec2-metadata --instance-id | cut -d ' ' -f 2)"
echo "Instance Type: $(ec2-metadata --instance-type | cut -d ' ' -f 2)"
echo "Availability Zone: $(ec2-metadata --availability-zone | cut -d ' ' -f 2)"
echo "=========================================="

# Environment variables from Terraform
S3_BUCKET="${s3_bucket_name}"
PROJECT_NAME="${project_name}"
AWS_REGION="${aws_region}"
AUTO_TERMINATE="${auto_terminate}"
ENABLE_TENSORBOARD="${enable_tensorboard}"

# Constants
WORKSPACE_DIR="/home/ubuntu/Clown_Fish_Acoustics"
CONDA_ENV="anemonefish_model"
TRAINING_SCRIPT="scripts/training/target_to_noise_classifier.py"

echo "Configuration:"
echo "  S3_BUCKET: $S3_BUCKET"
echo "  PROJECT_NAME: $PROJECT_NAME"
echo "  AWS_REGION: $AWS_REGION"
echo "  AUTO_TERMINATE: $AUTO_TERMINATE"
echo "  WORKSPACE_DIR: $WORKSPACE_DIR"
echo "=========================================="

# Wait for cloud-init to complete
echo "Waiting for cloud-init to complete..."
cloud-init status --wait

# Update system packages
echo "Updating system packages..."
apt-get update
apt-get install -y awscli jq tmux htop nvtop

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi || echo "WARNING: nvidia-smi not available"

# Initialize conda for bash
echo "Initializing conda..."
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f "/home/ubuntu/anaconda3/etc/profile.d/conda.sh" ]; then
    source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
elif [ -f "/home/ubuntu/miniconda3/etc/profile.d/conda.sh" ]; then
    source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
else
    echo "ERROR: Conda not found!"
    exit 1
fi

# Create conda environment
echo "Creating conda environment: $CONDA_ENV"
conda create -n $CONDA_ENV python=3.10 -y
conda activate $CONDA_ENV

# Verify Python version
echo "Python version: $(python --version)"

# Create workspace directory
echo "Creating workspace directory: $WORKSPACE_DIR"
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Download project files from S3
echo "Downloading project files from S3..."
aws s3 sync s3://$S3_BUCKET/code/ $WORKSPACE_DIR/ --region $AWS_REGION

# Download training data from S3
echo "Downloading training data from S3..."
mkdir -p $WORKSPACE_DIR/data/2_training_datasets/v2_biological
aws s3 sync s3://$S3_BUCKET/data/v2_biological/ $WORKSPACE_DIR/data/2_training_datasets/v2_biological/ --region $AWS_REGION

# Download preprocessing config from S3
echo "Downloading preprocessing config from S3..."
aws s3 cp s3://$S3_BUCKET/config/preprocessing_config_v2_biological.yaml $WORKSPACE_DIR/data/2_training_datasets/v2_biological/ --region $AWS_REGION || echo "Config not found, will be created by update script"

# Update workspace path in config file
echo "Updating workspace path in config file..."
CONFIG_FILE="$WORKSPACE_DIR/data/2_training_datasets/v2_biological/preprocessing_config_v2_biological.yaml"
if [ -f "$CONFIG_FILE" ]; then
    # Backup original config
    cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"
    
    # Replace workspace_base_path
    sed -i "s|workspace_base_path:.*|workspace_base_path: $WORKSPACE_DIR|g" "$CONFIG_FILE"
    echo "Updated config file:"
    grep "workspace_base_path" "$CONFIG_FILE"
else
    echo "WARNING: Config file not found at $CONFIG_FILE"
fi

# Install requirements
echo "Installing Python dependencies..."
if [ -f "$WORKSPACE_DIR/requirements-aws.txt" ]; then
    pip install -r $WORKSPACE_DIR/requirements-aws.txt
elif [ -f "$WORKSPACE_DIR/requirements.txt" ]; then
    pip install -r $WORKSPACE_DIR/requirements.txt
else
    echo "ERROR: No requirements file found!"
    exit 1
fi

# Install the anemonefish_acoustics package
echo "Installing anemonefish_acoustics package..."
if [ -f "$WORKSPACE_DIR/setup.py" ]; then
    cd $WORKSPACE_DIR
    pip install -e .
else
    echo "WARNING: setup.py not found, skipping package installation"
fi

# Create necessary directories
echo "Creating output directories..."
mkdir -p $WORKSPACE_DIR/models
mkdir -p $WORKSPACE_DIR/logs
mkdir -p $WORKSPACE_DIR/results

# Set permissions
chown -R ubuntu:ubuntu $WORKSPACE_DIR
chmod -R 755 $WORKSPACE_DIR

# Copy training runner script
echo "Setting up training runner script..."
cat > /home/ubuntu/training_runner.sh << 'SCRIPT_EOF'
${file("${path.module}/training_runner.sh")}
SCRIPT_EOF

# Make training runner executable
chmod +x /home/ubuntu/training_runner.sh

# Replace placeholders in training runner
sed -i "s|{{S3_BUCKET}}|$S3_BUCKET|g" /home/ubuntu/training_runner.sh
sed -i "s|{{AWS_REGION}}|$AWS_REGION|g" /home/ubuntu/training_runner.sh
sed -i "s|{{AUTO_TERMINATE}}|$AUTO_TERMINATE|g" /home/ubuntu/training_runner.sh
sed -i "s|{{WORKSPACE_DIR}}|$WORKSPACE_DIR|g" /home/ubuntu/training_runner.sh
sed -i "s|{{CONDA_ENV}}|$CONDA_ENV|g" /home/ubuntu/training_runner.sh
sed -i "s|{{TRAINING_SCRIPT}}|$TRAINING_SCRIPT|g" /home/ubuntu/training_runner.sh

# Start TensorBoard if enabled
if [ "$ENABLE_TENSORBOARD" = "true" ]; then
    echo "Starting TensorBoard..."
    mkdir -p $WORKSPACE_DIR/logs
    
    # Create TensorBoard systemd service
    cat > /etc/systemd/system/tensorboard.service << EOF
[Unit]
Description=TensorBoard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$WORKSPACE_DIR
Environment="PATH=/opt/conda/envs/$CONDA_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/opt/conda/envs/$CONDA_ENV/bin/tensorboard --logdir=$WORKSPACE_DIR/logs --host=0.0.0.0 --port=6006
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable tensorboard
    systemctl start tensorboard
    echo "TensorBoard started on port 6006"
fi

# Create training systemd service
echo "Creating training service..."
cat > /etc/systemd/system/training.service << EOF
[Unit]
Description=Anemonefish Training
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$WORKSPACE_DIR
ExecStart=/bin/bash /home/ubuntu/training_runner.sh
StandardOutput=append:/home/ubuntu/training.log
StandardError=append:/home/ubuntu/training.log
Restart=no

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the training service
systemctl daemon-reload
systemctl enable training
systemctl start training

echo "=========================================="
echo "EC2 Initialization Complete!"
echo "=========================================="
echo "Training service started. Check logs with:"
echo "  tail -f /home/ubuntu/training.log"
echo "  systemctl status training.service"
if [ "$ENABLE_TENSORBOARD" = "true" ]; then
    echo ""
    echo "TensorBoard is running on port 6006"
    echo "  Access via SSH tunnel: ssh -L 6006:localhost:6006 ubuntu@<instance-ip>"
    echo "  Then open: http://localhost:6006"
fi
echo "=========================================="
