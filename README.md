### Installation

Set up conda environment and install dependencies:

```bash
# Create a new environment
conda create --name MolMMP python==3.10
conda activate MolMMP

# Install PyTorch and core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Clone the repository
git clone https://github.com/HpuBioinformatics/MolMMP.git
cd MolMMP
Dataset
You can download the dataset file under https://github.com/ThomasSu1/SynthMol/tree/main/Data.



Training
To train MolMMP, configure the hyperparameters in model_train.py. Detailed descriptions of all configurable variables are provided in the source file.

# Navigate to the source directory
cd MolMMP

# Run the training script
$ python model_train.py

#Acknowledgements

ChemBERTa: https://huggingface.co/DeepChem/ChemBERTa-77M-MLM
Original implementation based on SynthMol project
