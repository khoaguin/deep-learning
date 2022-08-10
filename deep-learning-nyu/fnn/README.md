# Fully Connected Neural Network
This is my implementation (in Python) of a simple neural network with 2 linear layers (both the forward and backward pass). It is also the first homework of the course DS-GA 1008 (Deep Learning) at NYU, [here](https://atcold.github.io/NYU-DLSP21/) is the link to the course. `theory.pdf` contains my answers for the theory part in `homework1.pdf`.

The implementation part is in `mlp.py`. Based on the coure's homework, torch `Tensor` is used for convenience. Torch tensors are similar to numpy arrays, but tensors can be oprated on CUDA GPUs, and they can also keep the gradients of the loss function w.r.t themselves for automatic backward. Learn more about the differences between numpy arrays and torch tensors [here](https://medium.com/@ashish.iitr2015/comparison-between-pytorch-tensor-and-numpy-array-de41e389c213). PyTorch offers `autograd` functionality where we can define the forward pass to calculate the scalar tensor `loss`, then use `loss.backward()` to command the library to calculate the gradients for us. However, in this implementation, we will not use the torch's `autograd` but implement the backward pass ourselves. The testing files in `tests/` directory will compare our implementations with `torch`'s functions to see if the implementations are correct.  
The code in `training-CHD.ipynb` use the manually constructed network to train a simple 2-layer neural network to predict the risk of future Coronary Heart Disease (CHD) using the Framingham dataset.  
In addition to the default tests offered by the course homework, I also added a few new tests:
- `test_forward_mse.py`: compares the values of `y_hat`, `mse_loss`, and the derivatives of `mse_loss` with respect to `y_hat` of the forward pass of the manual neural net in `mlp.py` to the outputs of the same network using `torch`. Use this to test after you have implemented the forward pass before moving on to the backward pass.
- `test_forward_bce.py`: compares the values of `y_hat`, `bce_loss`, and the derivatives of `bce_loss` with respect to `y_hat` of the forward pass of the manual neural net in `mlp.py` to the outputs of the same network using `torch`. There is one problem when using `sigmoid` and `bce_loss`: when you increase the number of features in the linear layers, the results may become wrong as there are divisions by very small numbers. This is something that I need to investigate deeper.
- `test4.py`: test the derivatives of the mse loss function w.r.t the weights and biases for a neural network with one ReLU and one Sigmoid activation functions.
- `test5.py`: similar to `test4.py`, but for bce loss function.