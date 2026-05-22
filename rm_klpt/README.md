# Real Multiplication KLPT

This is an implementation of a generalisation of the KLPT algorithm to quaternion algebras over a totally real number field.
It is applied to the two dimensional version of the KLPT problem for principally polarised superspecial abelian surfaces with real multiplication.

## Running

This code was tested using Sage version 10.8 and Python version 3.14.4.
See the notebook `demo.ipynb` for an example of the computation.
The file `tests.py` may be loaded in sage in order to run some unit tests.

## File organisation

- `demo.ipynb` is a Jupyter notebook which presents some example usage of the klpt method.
- `rm_klpt/quadratic_form_nf.py` solves unbalanced diagonal quadratic forms defined over a number field.
- `rm_klpt/quaternion_ideal.py`  implements a class for fractional in a quaternion algebra over a number field.
- `rm_klpt/quaternion_matrix.py` implements a wrapper class for matrices over a rational quaternion algebra.
- `rm_klpt/random_walk.py` implements random walk in the RM 2-isogeny graph, and is used to generate example inputs.
- `rm_klpt/rm_data.py` implements the core of the klpt computation.
- `tests_klpt.py` contains unit tests for some functions.
