#!/bin/sh

cc_path="/Applications/CloudCompare.app/Contents/MacOS/CloudCompare"
scan="/Users/pabloaguilar/Documents/TFM_data/Sequence00_pcd"
echo "Running CSF on scan: $scan"

# $cc_path -O $scan -CSF -SCENES FLAT -PROC_SLOPE -CLOTH_RESOLUTION 0.5 -MAX_ITERATION 500 -CLASS_THRESHOLD 0.5 -EXPORT_GROUND

# shellcheck disable=SC2231
for f in $scan/*.pcd;
do echo "$f";
$cc_path -SILENT -C_EXPORT_FMT PCD -EXT pcd -O "$f" -CSF -SCENES FLAT -PROC_SLOPE -CLOTH_RESOLUTION 0.5 -MAX_ITERATION 500 -CLASS_THRESHOLD 0.5 -EXPORT_GROUND;
done