#!/usr/bin/env python3
"""
Upload training data, configuration, and source code to S3 bucket for AWS training.
"""

import os
import sys
import argparse
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm


def get_file_size(file_path):
    """Get file size in bytes."""
    return os.path.getsize(file_path)


def upload_file_with_progress(s3_client, file_path, bucket_name, s3_key):
    """Upload a single file to S3 with progress bar."""
    file_size = get_file_size(file_path)
    
    with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(file_path)) as pbar:
        s3_client.upload_file(
            str(file_path),
            bucket_name,
            s3_key,
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
        )


def upload_directory_to_s3(s3_client, local_dir, bucket_name, s3_prefix, exclude_patterns=None):
    """
    Upload a directory to S3 with progress tracking.
    
    Args:
        s3_client: Boto3 S3 client
        local_dir: Local directory path
        bucket_name: S3 bucket name
        s3_prefix: S3 key prefix
        exclude_patterns: List of patterns to exclude (e.g., ['.git', '__pycache__'])
    """
    if exclude_patterns is None:
        exclude_patterns = []
    
    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"WARNING: Directory not found: {local_dir}")
        return
    
    # Collect all files to upload
    files_to_upload = []
    for file_path in local_path.rglob('*'):
        if file_path.is_file():
            # Check if file should be excluded
            should_exclude = False
            for pattern in exclude_patterns:
                if pattern in str(file_path):
                    should_exclude = True
                    break
            
            if not should_exclude:
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
                files_to_upload.append((file_path, s3_key))
    
    if not files_to_upload:
        print(f"No files to upload from {local_dir}")
        return
    
    print(f"\nUploading {len(files_to_upload)} files from {local_dir} to s3://{bucket_name}/{s3_prefix}/")
    
    # Upload files with progress
    for file_path, s3_key in files_to_upload:
        try:
            upload_file_with_progress(s3_client, file_path, bucket_name, s3_key)
        except Exception as e:
            print(f"ERROR uploading {file_path}: {e}")


def upload_single_file_to_s3(s3_client, local_file, bucket_name, s3_key):
    """Upload a single file to S3."""
    local_path = Path(local_file)
    if not local_path.exists():
        print(f"WARNING: File not found: {local_file}")
        return
    
    print(f"\nUploading {local_file} to s3://{bucket_name}/{s3_key}")
    try:
        upload_file_with_progress(s3_client, local_path, bucket_name, s3_key)
        print(f"? Uploaded successfully")
    except Exception as e:
        print(f"ERROR uploading {local_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Upload training data and code to S3 for AWS training deployment'
    )
    parser.add_argument(
        '--bucket',
        required=True,
        help='S3 bucket name'
    )
    parser.add_argument(
        '--data-dir',
        default='/data/2_training_datasets/v2_biological',
        help='Local path to training data directory (default: /data/2_training_datasets/v2_biological)'
    )
    parser.add_argument(
        '--workspace-dir',
        default=None,
        help='Workspace directory (default: current working directory)'
    )
    parser.add_argument(
        '--config-file',
        default=None,
        help='Path to preprocessing config YAML file'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip uploading training data (useful for re-uploading only code)'
    )
    
    args = parser.parse_args()
    
    # Set workspace directory
    if args.workspace_dir:
        workspace_dir = Path(args.workspace_dir)
    else:
        # Assume script is in scripts/aws_deployment/ and workspace is 2 levels up
        workspace_dir = Path(__file__).parent.parent.parent
    
    print("=" * 60)
    print("AWS Training Data Upload")
    print("=" * 60)
    print(f"Bucket: {args.bucket}")
    print(f"Region: {args.region}")
    print(f"Workspace: {workspace_dir}")
    print(f"Data directory: {args.data_dir}")
    print("=" * 60)
    
    # Create S3 client
    try:
        s3_client = boto3.client('s3', region_name=args.region)
        
        # Verify bucket exists
        s3_client.head_bucket(Bucket=args.bucket)
        print(f"? Bucket '{args.bucket}' exists and is accessible")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"ERROR: Bucket '{args.bucket}' not found")
        elif error_code == '403':
            print(f"ERROR: Access denied to bucket '{args.bucket}'")
        else:
            print(f"ERROR: {e}")
        sys.exit(1)
    
    # 1. Upload training data
    if not args.skip_data:
        print("\n" + "=" * 60)
        print("1. Uploading Training Data")
        print("=" * 60)
        
        data_dir = Path(args.data_dir)
        if data_dir.exists():
            upload_directory_to_s3(
                s3_client,
                data_dir,
                args.bucket,
                'data/v2_biological',
                exclude_patterns=['.DS_Store', '._']
            )
        else:
            print(f"WARNING: Data directory not found: {data_dir}")
            print("Skipping data upload. You may need to upload data manually.")
    else:
        print("\n" + "=" * 60)
        print("1. Skipping Training Data Upload (--skip-data flag)")
        print("=" * 60)
    
    # 2. Upload preprocessing config
    print("\n" + "=" * 60)
    print("2. Uploading Preprocessing Config")
    print("=" * 60)
    
    if args.config_file:
        config_file = Path(args.config_file)
    else:
        # Try to find config file in data directory
        config_file = Path(args.data_dir) / 'preprocessing_config_v2_biological.yaml'
    
    if config_file.exists():
        upload_single_file_to_s3(
            s3_client,
            config_file,
            args.bucket,
            'config/preprocessing_config_v2_biological.yaml'
        )
    else:
        print(f"WARNING: Config file not found: {config_file}")
        print("Skipping config upload.")
    
    # 3. Upload source code
    print("\n" + "=" * 60)
    print("3. Uploading Source Code")
    print("=" * 60)
    
    src_dir = workspace_dir / 'src'
    if src_dir.exists():
        upload_directory_to_s3(
            s3_client,
            src_dir,
            args.bucket,
            'code/src',
            exclude_patterns=['.git', '__pycache__', '.pyc', '.DS_Store', '._']
        )
    else:
        print(f"WARNING: Source directory not found: {src_dir}")
    
    # 4. Upload scripts
    print("\n" + "=" * 60)
    print("4. Uploading Scripts")
    print("=" * 60)
    
    scripts_dir = workspace_dir / 'scripts'
    if scripts_dir.exists():
        upload_directory_to_s3(
            s3_client,
            scripts_dir,
            args.bucket,
            'code/scripts',
            exclude_patterns=['.git', '__pycache__', '.pyc', '.DS_Store', '._', 'aws_deployment']
        )
    else:
        print(f"WARNING: Scripts directory not found: {scripts_dir}")
    
    # 5. Upload requirements files
    print("\n" + "=" * 60)
    print("5. Uploading Requirements Files")
    print("=" * 60)
    
    # Try requirements-aws.txt first, then requirements.txt
    requirements_files = ['requirements-aws.txt', 'requirements.txt']
    uploaded_requirements = False
    
    for req_file_name in requirements_files:
        req_file = workspace_dir / req_file_name
        if req_file.exists():
            upload_single_file_to_s3(
                s3_client,
                req_file,
                args.bucket,
                f'code/{req_file_name}'
            )
            uploaded_requirements = True
            break
    
    if not uploaded_requirements:
        print("WARNING: No requirements file found!")
    
    # 6. Upload setup.py
    print("\n" + "=" * 60)
    print("6. Uploading Setup Files")
    print("=" * 60)
    
    setup_file = workspace_dir / 'setup.py'
    if setup_file.exists():
        upload_single_file_to_s3(
            s3_client,
            setup_file,
            args.bucket,
            'code/setup.py'
        )
    else:
        print(f"WARNING: setup.py not found: {setup_file}")
    
    print("\n" + "=" * 60)
    print("Upload Complete!")
    print("=" * 60)
    print(f"\nAll files uploaded to s3://{args.bucket}/")
    print("\nNext steps:")
    print(f"  1. Navigate to the terraform directory: cd terraform/")
    print(f"  2. Initialize Terraform: terraform init")
    print(f"  3. Review the plan: terraform plan")
    print(f"  4. Deploy: terraform apply")


if __name__ == '__main__':
    main()
