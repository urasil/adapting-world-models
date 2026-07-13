#!/bin/bash
#SBATCH -J holoassist_dl
#SBATCH -A MLMI-ua248-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=download_%j.log

set -euo pipefail

cd /home/ua248/rds/hpc-work/adapting_world_models/holoassist

echo "Starting download at $(date)"
echo "Working directory: $(pwd)"

wget -c https://hl2data.z5.web.core.windows.net/holoassist-data-release/video_compress.tar
wget -c https://hl2data.z5.web.core.windows.net/holoassist-data-release/data-annotation-trainval-v1_1.json
wget -c https://holoassist.github.io/label_files/data-splits-v1_2.zip

echo "Finished at $(date)"
ls -lh
