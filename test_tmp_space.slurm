#!/bin/bash
#SBATCH --job-name=test_tmp
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0:05:00
#SBATCH --output=logs/test_tmp_%j.out

echo "=== Checking /tmp space on compute node ==="
df -h /tmp
echo ""
echo "Testing write to /tmp..."
mkdir -p /tmp/test_dir_$$
dd if=/dev/zero of=/tmp/test_dir_$$/test_file bs=1M count=1000 2>&1
rm -rf /tmp/test_dir_$$
echo "✓ Successfully wrote/removed 1GB test file to /tmp"
