#!/bin/bash

#SBATCH --job-name="vqe_passive_energy"
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tuanduc.hoang@oist.jp
##SBATCH --output=out_files/%j.out
#SBATCH -p compute

# #SBATCH -C xeon
#SBATCH -c 1
#SBATCH -t 1-0
#SBATCH --mem=5G
#SBATCH --array=0-99

module load ruse

#cd $PWD/src/

#########################################
# run our program
#########################################

source /home/t/tuan-hoang/miniconda3/etc/profile.d/conda.sh
conda activate mlenv
ruse -s --label=${SLURM_ARRAY_TASK_ID} python rxx_vqe_noisy.py ${SLURM_ARRAY_TASK_ID}

#########################################
# save our results on Bucket
#########################################

#scp -r plots/* deigo:/bucket/BuschU/rike/marin/results/.

# Clean up: remove the directory on work with our input data
#rm -rf plots/*
#rm -rf data/*

