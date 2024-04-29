#!/bin/bash

# Define the base path for models and tokenizers
BASE_MODEL_PATH="/workingdir/aistudent/QuantLoss/QuantLoss/src/misc/model_saved"

# List of epoch cycles to run for each model
declare -a epochs=(10)

# Base command without the Embeddings_length option
# Embeddings_length will be added dynamically in the loop below
BASE_COMMAND="python evaluate_outlier.py --Token_max_length 512"

# List each model and its corresponding tokenizer
declare -A models=(
  #["bert-base-chinese"]="bert-base-chinese_tokenizer.pt"
  #["bert-base-multilingual-uncased"]="bert-base-multilingual-uncased_tokenizer.pt"
  # ["bert-base-uncased"]="bert-base-uncased_tokenizer.pt"
  ["bert-large-uncased"]="bert-large-uncased_tokenizer.pt"
  #["google-t5-t5-base"]="google-t5-t5-base_tokenizer.pt"
  #["google-t5-t5-large"]="google-t5-t5-large_tokenizer.pt"
  #["google-t5-t5-small"]="google-t5-t5-small_tokenizer.pt"
)

# Iterate through the epochs
for epoch in "${epochs[@]}"
do
  # Then iterate through the model-tokenizer pairs
  for model in "${!models[@]}"
  do
    model_path="$BASE_MODEL_PATH/$model.pt"
    tokenizer="${models[$model]}"
    tokenizer_path="$BASE_MODEL_PATH/$tokenizer"
    
    # Set Embeddings_length based on model
    if [[ "$model" == "bert-large-uncased" || "$model" == "google-t5-t5-large" ]]; then
      embeddings_length=1024
    elif [[ "$model" == "google-t5-t5-small" ]]; then
      embeddings_length=512
    else
      embeddings_length=768
    fi

    # Define result path including epoch
    result_path="./outlier_results/${model}_${epoch}_epoch"
    mkdir -p "$result_path"
    
    # Construct and run the command with dynamic Embeddings_length
    echo "Evaluating $model for $epoch epochs with Embeddings_length $embeddings_length..."
    command="$BASE_COMMAND --EpochCycle $epoch --Embeddings_length $embeddings_length --path \"$result_path\" --ModelPath \"$model_path\" --TokenizerPath \"$tokenizer_path\""
    eval $command
  done
done

echo "All evaluations complete."