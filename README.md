# PyTorch EMDLoss
PyTorch 1.0 implementation of the approximate Earth Mover's Distance

This is a PyTorch wrapper of CUDA code for computing an approximation to the Earth Mover's Distance loss.

Original source code can be found [here](https://github.com/fxia22/pointGAN/tree/74b6c432c5eaa1e0a833e755f450df2ee2c5488e/emd). This repository updates the code to be compatible with PyTorch 1.0 and extends the implementation to handle arbitrary dimensions of data.

**Limitations and Known Bugs:**
 - Double tensors must have <=11 dimensions while float tensors must have <=23 dimensions. This is due to the use of CUDA shared memory in the computation. This shared memory is limited by the hardware to 48kB.
- When handling larger point sets (M, N > ~2000), the CUDA kernel will fail. I think this is due to an overflow error in computing the approximate matching kernel. Any suggestions to fix this would be greatly appreciated. 
