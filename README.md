# wabbit-opt
This repo implements a kinematic optimization framework for flows arround moving objects.

## Cloning 

To clone this repo use:

    git clone https://github.com/adaptive-cfd/WABBIT-opt.git --recursive

Note that if you forgot the --recursive flag you can do:

    git submodule update --init

The --recursive is needed since other repos will be cloned in the LIB/ folder!

## Installation-Requirements

Since we make use of different softwares please make sure you
fullfil all requirements of

 + WABBIT: https://github.com/adaptive-cfd/WABBIT
 + install conda: https://docs.conda.io/en/latest/miniconda.html

## Installation:

Open terminal and go to the root directory of wabbit-opt and run:
    
    make
    make conda-env
    conda activate wabbit-opt

If you want to install wabbit-opt on a cluster without direct internet access
use the [conda-pack](https://conda.github.io/conda-pack/) package.