# Jensen_Tsallis_Divergence_for_Supervised_Classification_under_Data_Imbalance
 
The official code for the NeurIPS 2024 paper []()

## Environment Setup
Create a conda environment, activate it, and install additional pip packages:

```bash
conda env create -f env.yml
conda activate JTD
```
## Running Experiments
Please check programs/JSON_parameters to see the pre-configured experiments. Each JSON file contains all the simulations of the experiment, achieved by combining the parameter lists stored inside. The final models are stored in the model/ folder. In this folder, you can find the JSON files to reproduce paper's experimens. 

For synthetic signal experiment, execute:
```bash
python main.py --job-name MNIST_LeNet_5_hyperparameter_seeking
```
For h optimization over CHB01_01:
```bash
python main.py --job-name MNIST_LeNet_5_hyperparameter_seeking 
```

For execution of CHB01_03 experiment:
```bash
python main.py --job-name CIFAR10_resnet34_final_training
```
 
