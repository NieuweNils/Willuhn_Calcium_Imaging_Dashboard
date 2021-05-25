#!/bin/bash

docker build -f Dockerfile.dash -t willuhn_lab/dash .
docker build -f Dockerfile.jupyter -t willuhn_lab/jupyter .
docker-compose up -d db dash jupyterlab