# Learning Cortical Hierarchies with Temporal Hebbian Updates
This repository is the official implementation of the
Frontiers Computational Neuroscience submission 2023
"Learning Cortical Hierarchies with Temporal Hebbian Updates".

## Install Python packages
All the needed Python libraries can be installed with conda by running:
```
$ conda env create -f environment.yml
```

## Running the configs
You can define a config file and use the `run_config.py` script with 
the specified config file.
```
$ python3 run_config.py --config_module=configs.name_of_my_config
```
We provide the configurations used for all results in Table 1 of 
the paper.
If you wish to run for several random seeds you can do so with the 
following command (nb_seeds is defined in the config and specifies 
the amount of seeds to be run):
```
$ python3 run_seed_robustness.py --config_module=configs.name_of_my_config --nb_seeds=5
```


With the option 'save_correlations', you can save the correlation values 
between the current learning update and the DFC differential Hebbian updates. 
This allows you to reproduce the values used for Fig. 4.

With the option 'surprise_shuffle', you can reproduce the values used in Fig. 5.

With the option 'stdp_samples', you can increase the number fo conversion samples 
for from rates to spikes used for the STDP update.
This allows you to reproduce the values used for Fig. S1.

## References
This code base is built further upon the code base of 
Meulemans et al. 2020, "A Theoretical Framework for Target Propagation" and 
"Credit Assignment in Neural Networks through Deep Feedback Control", with the
Apache 2.0 license:
https://www.apache.org/licenses/LICENSE-2.0