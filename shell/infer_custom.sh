#!/bin/sh

./../bin/python \
./../main.py \
-d "/Users/pabloaguilar/Downloads/custom_dataset/lidar_frame" \
-w "/Users/pabloaguilar/Downloads/custom_dataset" \
-r "/Users/pabloaguilar/Downloads/custom_dataset/ranges" \
-m ./../pretrained \
-v no \
-l no \
-s custom
