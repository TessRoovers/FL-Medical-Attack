#!/bin/bash
# Set job requirements
#SBATCH --job-name TRAIN
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=32:00:00

cp $HOME/evaluate.py "$TMPDIR"
cp -r $HOME/unet "$TMPDIR"
cp -r $HOME/imgs "$TMPDIR"
cp -r $HOME/masks "$TMPDIR"
cp -r $HOME/utils "$TMPDIR"

mkdir -p "$TMPDIR"/output_dir

module load 2023
module load CUDA/12.1.1

source activate env2
echo "Activation complete.."

PROBS=("0.001" "0.005" "0.01" "0.02" "0.03" "0.04" "0.05")
STDS=("0.01" "0.02" "0.03" "0.04" "0.05" "0.1" "0.2")
PARAMS=("A" "B" "C" "D" "E" "F" "G")

for ((i=0; i<7; i++)); do
    PROB=${PROBS[i]}
    STD=${STDS[i]}
    PARAM=${PARAMS[i]}


    python3 -uc "import torch; print('PyTorch:\n GPU available...?', torch.cuda.is_available(), '\n PyTorch version:', torch.__version__)"
    echo "Running training code for $PARAM..."
    python3 "$HOME/train_gc.py" --prob $PROB --std $STD --param $PARAM &
    echo "Training code for $PARAM started!"

done
wait


echo "Finished training for parameters."

cp -r "$TMPDIR"/output_dir "$HOME"

echo "Output files copied to home folder."
echo "Job finished at $(date)."

conda deactivate
