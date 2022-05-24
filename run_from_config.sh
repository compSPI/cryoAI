#!/bin/bash
#SBATCH --partition=ml
#SBATCH --job-name=cryoai
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -n 32
#SBATCH --output=outputs/%j.log
#SBATCH --error=outputs/%j.err

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -c|--config)
      RELATVECONFIGPATH="$2"
      shift
      shift
      ;;
    --sif)
      CONTAINERPATH="$2"
      shift
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

echo "Relative path for config:"
echo ${RELATVECONFIGPATH}
echo "Absolute path for container:"
echo ${CONTAINERPATH}

singularity exec -B /sdf --nv ${CONTAINERPATH} \
            python -m src.reconstruct.main -c ${RELATVECONFIGPATH} --job_id ${SLURM_JOBID}
