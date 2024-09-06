# LightNet
<img src="https://github.com/parikshit-gupta/LightNet/blob/main/image.jpeg" width="1000"/>
A small and simple framework which implements the concept of computational graphs and reverse auto-diff with a neural network library on top of it with tensorflow/pytorch like API. It breaks each neural network into the mini operations that make it i.e. the dot product and the activation(LightNet only supports the sigmoid activation). All data and mini operations are stored in nodes of a dynamically built computational graph, these nodes knit together to build neurons, which inturn are knit to get layers and finally the layers are knit to get the neural network. It uses graphviz to visualise the computational grpahs.

LightNet is plausibly useful for educational purposes, can be used to implement and train neural nets over small and moderately large datasets (maxsize after tiling can be of the order 10^6) in a reasonable time frame. 

There definitely is scope of optimisation in the implementation, anyone intrigued enough should pursue it.
To dive deep into how LightNet was built step by step refer the Jupyter Notebook 'Reverse_autodiff_py.ipynb'.

LightNet gave the results expected off a neural network when tested on a toy dataset and the famous coffee roasting example. Check out the respective directories to see the tests.
