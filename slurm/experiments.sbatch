#!/bin/bash
# One Slurm array task per experiment in the settings file.
#
#   mkdir -p outputs/slurm                      # Slurm does not create the log folder
#   python experiments.py status                # lists the experiment indices and the --array range
#   sbatch --array=0-5 slurm/experiments.sbatch
#   SETTINGS=my_settings.json sbatch --array=0-3 slurm/experiments.sbatch
#
# Experiments with a valid saved result are skipped, so resubmitting the whole
# array after a failure or a code change only reruns what is needed.
#
# --cpus-per-task must be at least the largest "parallel_workers" in the
# settings file; the runner refuses an experiment that asks for more workers
# than the job was given. Adjust memory, time, and partition for your cluster.

#SBATCH --job-name=calibra
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate integrates_hackathon

python experiments.py run \
    --settings "${SETTINGS:-experiments.json}" \
    --index "${SLURM_ARRAY_TASK_ID}"
