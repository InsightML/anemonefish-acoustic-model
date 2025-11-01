#!/bin/bash
set -e

# Configuration (will be replaced by user_data.sh)
S3_BUCKET="{{S3_BUCKET}}"
AWS_REGION="{{AWS_REGION}}"
AUTO_TERMINATE="{{AUTO_TERMINATE}}"
WORKSPACE_DIR="{{WORKSPACE_DIR}}"
CONDA_ENV="{{CONDA_ENV}}"
TRAINING_SCRIPT="{{TRAINING_SCRIPT}}"

# Logging
LOGFILE="/home/ubuntu/training.log"
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "Starting Training Execution"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Configuration:"
echo "  S3_BUCKET: $S3_BUCKET"
echo "  AWS_REGION: $AWS_REGION"
echo "  WORKSPACE_DIR: $WORKSPACE_DIR"
echo "  CONDA_ENV: $CONDA_ENV"
echo "  TRAINING_SCRIPT: $TRAINING_SCRIPT"
echo "  AUTO_TERMINATE: $AUTO_TERMINATE"
echo "=========================================="

# Initialize conda
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

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
conda activate $CONDA_ENV

# Verify GPU is available
echo "Checking GPU availability..."
nvidia-smi

# Change to workspace directory
cd $WORKSPACE_DIR

# Run the training script
echo "=========================================="
echo "Starting Training Script"
echo "=========================================="
START_TIME=$(date +%s)

# Run training and capture exit code
set +e
python $TRAINING_SCRIPT
TRAINING_EXIT_CODE=$?
set -e

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "=========================================="
echo "Training Completed"
echo "=========================================="
echo "Exit Code: $TRAINING_EXIT_CODE"
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="

# Sync results to S3
echo "Syncing results to S3..."

# Sync models directory
if [ -d "$WORKSPACE_DIR/models" ]; then
    echo "Uploading models to S3..."
    aws s3 sync $WORKSPACE_DIR/models/ s3://$S3_BUCKET/results/models/ --region $AWS_REGION
    echo "Models uploaded successfully"
fi

# Sync logs directory
if [ -d "$WORKSPACE_DIR/logs" ]; then
    echo "Uploading logs to S3..."
    aws s3 sync $WORKSPACE_DIR/logs/ s3://$S3_BUCKET/results/logs/ --region $AWS_REGION
    echo "Logs uploaded successfully"
fi

# Upload training log
if [ -f "/home/ubuntu/training.log" ]; then
    echo "Uploading training log to S3..."
    aws s3 cp /home/ubuntu/training.log s3://$S3_BUCKET/results/training.log --region $AWS_REGION
fi

# Upload initialization log
if [ -f "/home/ubuntu/init.log" ]; then
    echo "Uploading initialization log to S3..."
    aws s3 cp /home/ubuntu/init.log s3://$S3_BUCKET/results/init.log --region $AWS_REGION
fi

echo "=========================================="
echo "Results Upload Complete"
echo "=========================================="
echo "Results available at:"
echo "  Models: s3://$S3_BUCKET/results/models/"
echo "  Logs: s3://$S3_BUCKET/results/logs/"
echo "  Training Log: s3://$S3_BUCKET/results/training.log"
echo "=========================================="

# Auto-terminate instance if enabled
if [ "$AUTO_TERMINATE" = "true" ]; then
    echo "Auto-termination is enabled"
    echo "Getting instance ID..."
    INSTANCE_ID=$(ec2-metadata --instance-id | cut -d ' ' -f 2)
    echo "Instance ID: $INSTANCE_ID"
    
    echo "Waiting 60 seconds before termination to ensure S3 sync completion..."
    sleep 60
    
    echo "Terminating instance..."
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $AWS_REGION
    echo "Instance termination initiated"
else
    echo "Auto-termination is disabled. Instance will continue running."
    echo "Remember to manually terminate the instance to avoid unnecessary costs!"
fi

echo "=========================================="
echo "Training Runner Complete"
echo "=========================================="
