### Predictive Coding in Python

## <img alt="logo" src="https://www.frontiersin.org/files/Articles/18458/fpsyg-02-00395-r3/image_m/fpsyg-02-00395-g003.jpg" height="180"> 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

A `Python` implementation of _An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity_

[[Paper](https://www.mrcbndu.ox.ac.uk/sites/default/files/pdf_files/Whittington%20Bogacz%202017_Neural%20Comput.pdf)]

Based on the `MATLAB` [implementation](https://github.com/djcrw/Supervised-Predictive-Coding) from [`@djcrw`]

## Requirements
- `numpy`
- `torch`
- `torchvision` 


Modified by Pau Vilimelis Aceituno

Includes modifications to use differential hebbian learning on lines 154-229 of network.py and line 103 on script.py. The use of differential hebbian can be turned off by setting cf.DH to false