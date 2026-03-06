#!/usr/bin/env bash

# This script just zip the src folder into a OlympiadLevelMaths4LLM.zip file for easy upload to Kaggle.
# ignore the __pycache__ and .git folders.

zip -r OlympiadLevelMaths4LLM.zip src/ scripts/ \
    -x "*.pyc" \
    -x "__pycache__/*" \
    -x "*.pyo" \
    -x "*.pyd" \
    -x "*.git/*" \
    -x "*.idea/*" \
    -x "*.vscode/*" \
    -x "*.DS_Store"

echo "Created OlympiadLevelMaths4LLM.zip"