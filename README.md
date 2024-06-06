# A time series feature extraction methodology based on multiscale overlapping windows, adaptive KDE, and entropic and information differentiable functionals

## Environment Setup
Create a conda environment, activate it, and install additional pip packages:

```bash
conda env create -f env_TDE.yml
conda activate env_TDE
```
## Running Experiments
Please check programs/JSON_parameters to see the pre-configured experiments. Each JSON file contains all the simulations of the experiment, achieved by combining the parameter lists stored inside. The final models are stored in the model/ folder. In this folder, you can find the JSON files to reproduce paper's experimens. 

For synthetic signal experiment, execute:
```bash
python main.py KDE_synthetic_signal_case
```

For h optimization over CHB01_01:
```bash
python main.py KDE_EEG_chb1_01
```

For execution of CHB01_03 experiment:
```bash
python main.py KDE_EEG_chb1_03_seizure
```
 
