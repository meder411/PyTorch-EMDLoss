# PyTorch EMDLoss
PyTorch 1.0 implementation of the approximate Earth Mover's Distance

This is a PyTorch wrapper of CUDA code for computing an approximation to the Earth Mover's Distance loss.

Original source code can be found [here](https://github.com/fxia22/pointGAN/tree/74b6c432c5eaa1e0a833e755f450df2ee2c5488e/emd). This repository updates the code to be compatible with PyTorch 1.0 and extends the implementation to handle arbitrary dimensions of data.

Installation should be as simple as running `python setup.py install`.

**Limitations and Known Bugs:**

Bugs is repaired.

Success work in (Ubuntu 16.04, Nvidia-Driver 410, CUDA9.0, CUDNN7.1.4)
