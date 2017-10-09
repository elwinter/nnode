# nnode

Neural network code for solving ordinary differential equations

This code is my first attempt at implementing the Lagaris et al method
for using neural networks to solve differential equations.

Reference: Lagaris et al, IEEE Transactions on Neural Networks 8(5),
p. 987 (1998

The main code is in nnode.py. Run 'nnode.py --help' for a description
of usage.

The files named 'analytical*' are the Mathematica (.nb) and Jupyter
(.ipynb) notebooks used for exploring the specified ODE, and to ensure
that Python code gives the same results as Mathematica code.

The files 'ode*.py' are the python modules which implement the code
required for the respective ODEs.

The files ending in '00' are for the simple ODE used during
development.

The files ending in '01' are for the first ODE in Lagaris et al
(equation (27)).

The files ending in '02' are for the second ODE in Lagaris et al
(equation (28)).  )

The file sigma.py provides code for the sigmoid transfer function and
its derivatives. The corresponding .nb and .ipynb files are Jupter and
Mathematica notebooks for examination of the sigmoid function, and to
ensure the Python code gives the same results as Mathematica.

The files scratch.* are for temporary work and testing (.nb for
Mathematica, .ipynb for Jupyter).

Eric Winter
ewinter@stsci.edu
2017-10-09
