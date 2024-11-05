#!/bin/bash

#SBATCH -A berzelius-2024-329  # Replace with your project account
#SBATCH --gpus=2                   # Request 5 GPUs
#SBATCH -C "fat"
#SBATCH --time=30:00:00            # Set the time limit (12 hours here)

# scp /Users/niklas/Documents/code_projects/SQLDescriptionGeneration/data.zip x_nikwr@berzelius.nsc.liu.se:/proj/berzelius-aiics-real/users/x_nikwr/SQLDescriptionGeneration/

export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_nikwr/SQLDescriptionGeneration/.huggingface_cache" #Important so that HuggingFace models or datasets are not saved in home dir
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_TOKEN="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"
export OPENAI_API_KEY="sk-proj-E61k4N2SHy2VMUIAIEXHT3BlbkFJQ2AmGFKd3RDZjqPiqEhM"

# Pass the number of GPUs to the Python script
apptainer exec --nv env.sif python3 text_to_sql_all_schemas_new.py