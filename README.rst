geneticalgorithm2
=================


geneticalgorithm2 is a Python library distributed on
`Pypi <https://pypi.org>`__ for implementing standard and elitist
`genetic-algorithm <https://towardsdatascience.com/introduction-to-optimization-with-genetic-algorithm-2f5001d9964b>`__
(GA). This package solves continuous,
`combinatorial <https://en.wikipedia.org/wiki/Combinatorial_optimization>`__
and mixed
`optimization <https://en.wikipedia.org/wiki/Optimization_problem>`__
problems with continuous, discrete, and mixed variables. It provides an
easy implementation of genetic-algorithm (GA) in Python.

Installation / PLEASE CHECK THE HOMEPAGE FOR CURRENT EXAMPLES OF USING
----------------------------------------------------------------------

Use the package manager `pip <https://pip.pypa.io/en/stable/>`__ to
install geneticalgorithm2 in Python.

.. code:: python

    pip install geneticalgorithm2

A simple example
----------------

Assume we want to find a set of X=(x1,x2,x3) that minimizes function f(X)=x1+x2+x3 where X can be any real number in [0,10].
This is a trivial problem and we already know that the answer is X=(0,0,0) where f(X)=0. We just use this simple example to see how to implement geneticalgorithm:

First we import geneticalgorithm and `numpy <https://numpy.org>`__.
Next, we define function f which we want to minimize and the boundaries
of the decision variables; Then simply geneticalgorithm is called to
solve the defined optimization problem as follows:

.. code:: python

    import numpy as np
    from geneticalgorithm2 import geneticalgorithm2 as ga

    def f(X):
        return np.sum(X)
        
        
    varbound=np.array([[0,10]]*3)

    model=ga(function=f,dimension=3,variable_type='real',variable_boundaries=varbound)

    model.run()

Notice that we define the function f so that its output is the objective
function we want to minimize where the input is the set of X (decision
variables). The boundaries for variables must be defined as a numpy
array and for each variable we need a separate boundary. Here I have
three variables and all of them have the same boundaries (For the case
the boundaries are different see the example with mixed variables).
