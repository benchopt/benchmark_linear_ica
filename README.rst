Benchopt Benchmark for Linear ICA
=================================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to Independent Component Analysis (ICA):

$$X = A S$$

where

$$X \\in \\mathbb{R}^{p \\times n}, A \\in \\mathbb{R}^{p \\times p} \\text{ and } S \\in \\mathbb{R}^{p \\times n}$$

such that $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features.
The purpose of linear ICA is to recover the mixing matrix $A$ from $X$.

where X is n_features x n_samples, A is n_features x n_features and S
is n_features x n_samples. The purpose of ICA is to recover the
mixing matrix A from X.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_linear_ica
   $ benchopt run benchmark_linear_ica

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_linear_ica -s fastica -d simulated --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_linear_ica/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_linear_ica/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
