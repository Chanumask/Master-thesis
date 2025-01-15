#!/usr/bin/env bash
#SBATCH --job-name=vqe-dropout
#SBATCH --comment="Solving Max-Cut with VQE"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.friedrich@campus.lmu.de
#SBATCH --partition=All
#SBATCH --nodes 1
#SBATCH -B 1:7:2
#SBATCH -n 1

SLURM_DIR="slurm"
ERROR_LOG="$SLURM_DIR/error.log"  # Error log file in the slurm folder

# Create the slurm directory if it doesn't exist
mkdir -p $SLURM_DIR

# Activate python environment
source pyenv/bin/activate && echo "pyenv loaded"

# Absolute path to the Python script (ensure it is correct)
PYTHON_SCRIPT="/home/f/friedrichj/Desktop/vqe/Master-thesis/main.py"

# Run the python job and capture both stdout and stderr inside the slurm folder
echo "Running Python job and capturing output and error..."
srun --ntasks 1 --nodes=1 -c 14 python3 "$PYTHON_SCRIPT" > "$SLURM_DIR/output.txt" 2> "$ERROR_LOG"

# Check if any error occurred and store it in the error log and email body
echo "Checking for errors in the error log..."

# Ensure the error file has content
if [ -s "$ERROR_LOG" ]; then
    #If error log has content, include it in the email body
    echo "Job completed with errors. See the error trace below:" > "$SLURM_DIR/email_body.txt"
    cat "$ERROR_LOG" >> "$SLURM_DIR/email_body.txt"
else
    # If no errors, simply notify success
    echo "Job completed successfully. See the output below:" > "$SLURM_DIR/email_body.txt"
    cat "$SLURM_DIR/output.txt" >> "$SLURM_DIR/email_body.txt"
fi

# move all output files to the slurm folder
echo "Moving .out files to SLURM_DIR"
mv *.out "$SLURM_DIR"

# Send the results via email with any errors included in the email body
echo "Sending email with very nice results."

# The '-i' flag points to the email body file and '-a' attaches the zip file
echo "Please find the job results and any error messages below." | mutt -s "QCP Train Routing Results" -a "$SLURM_DIR" -i "$SLURM_DIR/email_body.txt" -- j.friedrich@campus.lmu.de

echo "Slurm execution done."
