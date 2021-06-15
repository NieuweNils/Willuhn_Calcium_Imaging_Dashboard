#!/bin/bash

docker build -f Dockerfile.dash -t willuhn_lab/dash ../dash
#docker build -f Dockerfile.jupyter -t willuhn_lab/jupyter ../jupyter
docker-compose up -d db dash