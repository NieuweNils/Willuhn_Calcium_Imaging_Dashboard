#!/bin/bash

docker build -t willuhn_lab/mysql-pxb:5.6 .
docker run -it willuhn_lab/mysql-pxb:5.6
