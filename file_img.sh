#!/bin/bash


img_dir1="/mnt/afs/users/konglingshun/dataset/Train_dataset_single_kernel_127_anxiety_001_without_sam/HR/"
img_dir2="/mnt/afs/users/konglingshun/dataset/data1_patch/HR/"
img_dir3="/mnt/afs/users/konglingshun/dataset/data3_patch/HR/"

output_file="files_Image.list"

> "$output_file"

temp_file=$(mktemp)

for img_dir in "$img_dir1" "$img_dir2" "$img_dir3"; do
  echo "Searching files in: $img_dir"
  find "$img_dir" -type f >> "$temp_file"
done

shuf "$temp_file" > "$output_file"

rm "$temp_file"

echo "Files from all directories have been randomly listed in $output_file"