#!/bin/bash

# Configuration variables
SERVER_USER="insightml"
SERVER_IP="192.168.1.200"
SERVER_CONNECTION="${SERVER_USER}@${SERVER_IP}"
IMAGE_NAME="anemonefish_acoustics"
TAR_FILE="${IMAGE_NAME}.tar"
REMOTE_PATH="/home/${SERVER_USER}/"

docker save ${IMAGE_NAME} -o ${TAR_FILE}

scp ${TAR_FILE} ${SERVER_CONNECTION}:${REMOTE_PATH}

ssh ${SERVER_CONNECTION} "docker load -i ${REMOTE_PATH}${TAR_FILE}"

