# nnode

Neural network code for solving ordinary differential equations

This code is my first attempt at implementing the Lagaris et al method
for using neural networks to solve differential equations.

Reference: Lagaris et al, IEEE Transactions on Neural Networks 8(5),
p. 987 (1998

Help for all programs is available using the ih command-line option.

The program nnode1.py is used to solve 1st-order ODE IVP. This program
can process ode00.py, lagaris01.py, and lagaris02.py.

The program nnode2.py is used to solve 2nd-order ODE BVP. This program
can process lagaris03bvp.py.

The program nnode3.py is used to solve 2nd-order ODE IVP. This program
can process lagaris03ivp.py.

The files lagaris01.py, lagaris02.py define the code needed for
problems 1 and 2 in the Lagaris et al paper. The files lagaris03bvp.py
and lagaris03ivp.py define the BVP and IVP versions of Problem 3 in
Lagaris et al. The file ode00.py is a simple 1st order ODE problem for
testing nnode1.py.

The file sigma.py provides code for the sigmoid transfer function and
its derivatives. The corresponding .nb and .ipynb files are Jupyter
and Mathematica notebooks for examination of the sigmoid function, and
to ensure the Python code gives the same results as Mathematica.

The file nnode1.ipynb is a detailed walkthrough of nnode1.py,
explaining the algorithm and program options using several examples.

The remaining iPython (*.ipynb) and Mathematica (*.nb) files are
notebooks used for ensuring the Python and Mathematica code gives the
same results.

Eric Winter
ewinter@stsci.edu
2017-10-31
