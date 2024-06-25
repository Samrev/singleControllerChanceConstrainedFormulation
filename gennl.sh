#!/bin/bash
set -x
models_directory="./models"
data_directory="./test_cases"
nl_directory="./nlmodels/"

# Checking if the model_directory is not a directory
if [ ! -d "$models_directory" ]; then
  echo "model_directory is not a directory"
  exit 1
fi

# Loop through models in the model_directory
for model in $models_directory/*; do
  if [[ $model == *.mod ]]; then
    # Loop through data's in the data_directory
    for data in $data_directory/*; do
      if [[ $data == *.dat ]]; then
        full_model_name="${model##*/}"
        model_name="${full_model_name%.mod}"

        full_data_name="${data##*/}"
        data_name="${full_data_name%.dat}"
        ampl ampl_script.run
        ampl -og"$nl_directory""$model_name"_"$data_name" "$model" "$data"
      fi
    done
  fi
done
