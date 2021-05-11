#!/bin/bash

docker build -t willuhn_lab/dash .
docker-compose up -d db dash jupyterlab