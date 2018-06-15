#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=5:00:00
#SBATCH --account=rrg-marchand
#SBATCH --array=0-{ntasks}
#SBATCH --output=test-%J.out
date
SECONDS=0
cd $HOME/projects/CFS/
python {main_file} --n $SLURM_ARRAY_TASK_ID
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date
