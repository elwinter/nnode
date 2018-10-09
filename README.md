# nnode

Neural network code for solving ordinary and partial differential equations.

Reference: Lagaris et al, IEEE Transactions on Neural Networks 8(5), p. 987 (1998).

## Modules

After cloning this repository, add the code directories to your PYTHONPATH:

* `bash`: `export PYTHONPATH=/path/to/nnode/eq:/path/to/nnode:$PYTHONPATH`

* `tcsh`: `setenv PYTHONPATH /path/to/nnode/eq:/path/to/nnode:$PYTHONPATH`

*NOTE*: This code assumes the user has a standard Anaconda 3-based environment. Other than the code in this package, no additional Python modules are required.

The following modules are provided:

* `nnode1ivp.py` - Use a neural network to solve 1st-order ODE IVP.

* `nnode2bvp.py` - Use a neural network to solve 2nd-order ODE BVP.

* `nnode2ivp.py` - Use a neural network to solve 2nd-order ODE IVP.

* `nnpde1ivp.py` - Use a neural network to solve 1st-order PDE IVP.

* `nnpde2bvp.py` - Use a neural network to solve 2nd-order PDE BVP.

* `nnpde2ivp.py` - Use a neural network to solve 2nd-order PDE IVP.

* `nnpde2diff1d.py` - Use a neural network to solve the 1-D diffusion equation, with time-varying BC.

The `eq` subdirectory contains a variety of sample equations to solve. The equations should be used with the following modules (files not listed are not guaranteed to work):

* `nnode1ivp.py` - `lagaris_01.py`, `lagaris_02.py`, `ode1_00.py`, `ode1_01.py`, `ode1_02.py`, `ode1_03.py`, `ode1_04.py`

* `nnode2bvp.py` - `lagaris_03_bvp.py`, `ode2_bvp_00.py`

* `nnode2ivp.py` - `lagaris_03_ivp.py`, `ode2_ivp_00.py`

* `nnpde1ivp.py` - `pde1_ivp_00.py`

* `nnpde2bvp.py` - `lagaris05.py`

* `nnpde2ivp.py` - No examples available

* `nnpde2diff1d.py` - `diff1d_*`

The `experiments` subdirectory contains another `README.md`, along with a set of Jupyter notebooks that illustrate the use of the `nnpde2diff1d.py` module. Notebooks for other modules will be added in the future.

## Contact
Eric Winter

ewinter@stsci.edu

2018-10-09
