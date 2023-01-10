#!/bin/sh

./../bin/python \
./../main.py \
-d "/Volumes/My Passport/data/dataset" \
-o "/Volumes/My Passport/data/dataset/sequences/00/predictions" \
-m ./../pretrained \
-v no \
-l no \
-s kitti
