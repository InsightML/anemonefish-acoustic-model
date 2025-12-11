#!/bin/bash

# Remove macOS extended attributes and hidden files
find . -path "./data" -prune -o -name "._*" -delete
xattr -cr .

# Build Docker image
docker build --platform linux/amd64 -t anemonefish_acoustics:latest .

# Tag the image for AWS ECR
docker tag anemonefish_acoustics:latest 944269089535.dkr.ecr.eu-west-2.amazonaws.com/anemonefish-acoustics:latest

# Push the image to AWS ECR
docker push 944269089535.dkr.ecr.eu-west-2.amazonaws.com/anemonefish-acoustics:latest