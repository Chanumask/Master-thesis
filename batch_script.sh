#!/usr/bin/env bash
#SBATCH --job-name=vqe-dropout
#SBATCH --comment="Solving Max-Cut with VQE"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.friedrich@campus.lmu.de
#SBATCH --partition=All
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH -n 1
#SBATCH -c 14

PYTHON_SCRIPT="/home/f/friedrichj/Desktop/vqe/Master-thesis/main.py"
SLURM_DIR="slurm"
ERROR_LOG="$SLURM_DIR/error.log"  # Error log file in the slurm folder
OUTPUT_LOG="$SLURM_DIR/output.txt"

# Create the slurm directory if it doesn't exist
mkdir -p $SLURM_DIR

# Activate python environment
source /home/f/friedrichj/Desktop/vqe/Master-thesis/venv/bin/activate && echo "Python environment activated"

# Absolute path to the Python script (ensure it is correct)
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Python script not found: $PYTHON_SCRIPT" > "$ERROR_LOG"
    exit 1
fi

# Run the python job and capture both stdout and stderr inside the slurm folder
echo "Running Python job..."
srun python3 "$PYTHON_SCRIPT" > "$OUTPUT_LOG" 2> "$ERROR_LOG"

# Check if any error occurred and store it in the error log and email body
echo "Checking for errors in the error log..."

# Ensure the error file has content
if [ -s "$ERROR_LOG" ]; then
    echo "Job completed with errors. See the error trace below:" > "$SLURM_DIR/email_body.txt"
    cat "$ERROR_LOG" >> "$SLURM_DIR/email_body.txt"
else
    echo "Job completed successfully. See the output below:" > "$SLURM_DIR/email_body.txt"
    cat "$OUTPUT_LOG" >> "$SLURM_DIR/email_body.txt"
fi

# move all output files to the slurm folder
echo "Moving SLURM .out files..."
mv *.out "$SLURM_DIR" 2>/dev/null
zip -r "$SLURM_DIR/output.zip" "$SLURM_DIR"

# Send the results via email with any errors included in the email body
if command -v mutt &> /dev/null; then
    echo "Please find the job results and any error messages below." | mutt -s "VQE entanglement study" -a "$SLURM_DIR/output.zip" -i "$SLURM_DIR/email_body.txt" -- j.friedrich@campus.lmu.de
else
    echo "mutt not found. Email notification skipped."
fi

echo "Slurm execution done."
