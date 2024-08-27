## Code structure

The main code for implementing prediction can be found in the model/ folder.
load_data contains the code for loading data from csv and pk files.
data_preprocessing contains the normalization steps before input to the model.
modelstructure contains the chi_model architecture.
utils contains a few utility functions, including model saving and loading.

## Demo data

Test smiles strings input can be found in demo_data folder.

## Dependencies

The sample code is run on a workstation with RTX4090 with the dependencies listed in environment.yml file.
To install the dependencies listed in the `environment.yml` file, you can use the following command:

```bash
conda env create -f environment.yml
```

This command will create a new conda environment with the specified dependencies. Make sure you have Anaconda or Miniconda installed before running this command.

Once the environment is created, you can activate it using:

```bash
conda activate <environment_name>
```

Replace `<environment_name>` with the name of the environment specified in the `environment.yml` file.

