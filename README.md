# COmodel
Modeling CO emission in late-time spectra.

## Requirements

  1. The usual:  `scipy`, `numpy`, `matplotlib`
  2. The less usual:  `emcee` and `corner`. These can be installed using conda:
    conda install -c conda_forge emcee corner

## Usage

First, you'll need Peter's models. You can get them [here](https://drive.google.com/drive/folders/1nOA85TVRBEYh2PNelMEqgiqscpE65iT0?usp=sharing).
Then, you run the `makeGrid` script, which will load the models into a `numpy` array suitable for using in `python`.

You can run the script direction. There's help with the `-h` flag:

  python COmodel.py -h
  
  
