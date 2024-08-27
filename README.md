# MTL_ChiParameter
Sample code for the paper titled "Predicting polymer-solvent miscibility using machine-learned Flory-Huggins interaction parameters" (https://pubs.acs.org/doi/10.1021/acs.macromol.2c02600). Please cite this paper if you have benefited from any data or codes provided in this study.

UPDATE on 2023.04.28: Our calculation data has been updated to include actual calculated values.

UPDATE on 2024.03.27: We have made the COSMO-RS calculation data open through figshare (https://figshare.com/articles/dataset/Flory-Huggins_interaction_parameter_calculated_by_COSMO-RS_simulation/25448056).

## Sample data

In the folder of sample_data, we provided the full data set of experimental Chi parameter values (data_Chi.csv) extracted from the supplementary table of Orwoll and Arnold, Physical Properties of Polymers Handbook, 2nd ed., Springer: New York, USA, 2007. We also included the manual classification labels for the polymers of each data point. The SMILES of polymers and solvents in the data column 'ps_pair' were joined by '_'. Due to data sharing restriction by the PoLyInfo database, we are not allowed to share those data openly in this repository. For the binary solubility labels, we provide an artificial set of data (data_PI.csv), where the content is simply a copy of data_Chi.csv with solubility defined as the Chi values being smaller than 0.5. For the computational Chi parameter data (data_COSMO.csv), we only included calculation data of the polymer-sovlent pairs that appeared in data_Chi.csv. The descriptors of each data set were also shared (desc_XX.csv).

WARNING: The artificial data of PoLyInfo labels are only for execution of the sample codes. These are not REAL DATA!

## Sample code

Although we only provide training data that is partially artificial, all user parameters and settings in the sample code were exactly the same as when we ran the code to produce the results in the paper for a 3-task model case. Readers can follow the code to understand the details of our model construction and training procedure. There are descriptions on the parameters inside the sample code to help user try different settings, including training with one, two or three tasks, and to use linear or quadratic temperature dependence model for predicting Chi values.

WARNING: The trained model using this sample code would not be directly comparable to the results in our paper because the PoLyInfo classification and calculated Chi data are missing.

NOTE: We also provide a separated file of sample code for calculating the descriptors that are organized in the same format as the descriptor data stored in the sample_data folder.

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

