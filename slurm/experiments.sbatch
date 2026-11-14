#!/bin/bash
# One Slurm array task per experiment in the settings file.
#
#   mkdir -p outputs/slurm                      # Slurm does not create the log folder
#   python experiments.py status                # lists the experiment indices and the --array range
#   sbatch slurm/experiments.sbatch             # all 49 experiments of experiments.json
#   sbatch --array=1-2,7-8 slurm/experiments.sbatch   # only some indices
#   SETTINGS=my_settings.json sbatch --array=0-3 slurm/experiments.sbatch
#
# The default --array below matches experiments.json (indices 0-48). Update it
# when experiments are added or removed, or pass --array on the command line,
# which overrides it.
#
# Experiments with a valid saved result are skipped, so resubmitting the whole
# array after a failure or a code change only reruns what is needed.
#
# --cpus-per-task must be at least the largest "parallel_workers" in the
# settings file; the runner refuses an experiment that asks for more workers
# than the job was given. Adjust memory, time, and partition for your cluster.

#SBATCH --job-name=calibra
#SBATCH --array=0-51
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --partition=cpu
#SBATCH --account=mtdlop

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source .venv/bin/activate

python experiments.py run \
    --settings "${SETTINGS:-experiments.json}" \
    --index "${SLURM_ARRAY_TASK_ID}"
