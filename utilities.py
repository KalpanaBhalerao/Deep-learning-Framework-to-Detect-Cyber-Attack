class loss:
    traning_loss = 9
"""
NumPy
=====

Provides
  1. An array object of arbitrary homogeneous items
  2. Fast mathematical operations over arrays
  3. Linear Algebra, Fourier Transforms, Random Number Generation

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide, available from
`the NumPy homepage <https://numpy.org>`_.

We recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `numpy` has been imported as ``np``::

  >>> import numpy as np

Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(np.sort)
  ... # doctest: +SKIP

For some objects, ``np.info(obj)`` may provide additional help.  This is
particularly true if you see the line "Help on ufunc object:" at the top
of the help() page.  Ufuncs are implemented in C, not Python, for speed.
The native Python help() does not know how to view their help, but our
np.info() function does.

To search for documents containing a keyword, do::

  >>> np.lookfor('keyword')
  ... # doctest: +SKIP

General-purpose documents like a glossary and help on the basic concepts
of numpy are available under the ``doc`` sub-module::

  >>> from numpy import doc
  >>> help(doc)
  ... # doctest: +SKIP

Available subpackages
---------------------
lib
    Basic functions used by several sub-packages.
random
    Core Random Tools
linalg
    Core Linear Algebra Tools
fft
    Core FFT routines
polynomial
    Polynomial tools
testing
    NumPy testing tools
distutils
    Enhancements to distutils with support for
    Fortran compilers support and more  (for Python <= 3.11).

Utilities
---------
test
    Run numpy unittests
show_config
    Show numpy build configuration
matlib
    Make everything matrices.
__version__
    NumPy version string

Viewing documentation using IPython
-----------------------------------

Start IPython and import `numpy` usually under the alias ``np``: `import
numpy as np`.  Then, directly past or use the ``%cpaste`` magic to paste
examples into the shell.  To see which functions are available in `numpy`,
type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use
``np.*cos*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to narrow
down the list.  To view the docstring for a function, use
``np.cos?<ENTER>`` (to view the docstring) and ``np.cos??<ENTER>`` (to view
the source code).

Copies vs. in-place operation
-----------------------------
Most of the functions in `numpy` return a copy of the array argument
(e.g., `np.sort`).  In-place versions of these functions are often
available as array methods, i.e. ``x = np.array([1,2,3]); x.sort()``.
Exceptions to this rule are documented.

"""
from sklearn import svm
from random import Random
"""
The :mod:`sklearn` module includes functions to configure global settings and
get information about the working environment.
"""

# Machine learning module for Python
# ==================================
#
# sklearn is a Python module integrating classical machine
# learning algorithms in the tightly-knit world of scientific Python
# packages (numpy, scipy, matplotlib).
#
# It aims to provide simple and efficient solutions to learning problems
# that are accessible to everybody and reusable in various contexts:
# machine-learning as a versatile tool for science and engineering.
#
# See http://scikit-learn.org for complete documentation.
def SequentialModel(layers):
    #
# sklearn is a Python module integrating classical machine
# learning algorithms in the tightly-knit world of scientific Python
# packages (numpy, scipy, matplotlib).
#
# It aims to provide simple and efficient solutions to learning problems
# that are accessible to everybody and reusable in various contexts:
# machine-learning as a versatile tool for science and engineering.
#
    _msg = (
        "module 'numpy' has no attribute '{n}'.\n"
        "`np.{n}` was a deprecated alias for the builtin `{n}`. "
        "To avoid this error in existing code, use `{n}` by itself. "
        "Doing this will not modify any behavior and is safe. {extended_msg}\n"
        "The aliases was originally deprecated in NumPy 1.20; for more "
        "details and guidance see the original release note at:\n"
        "    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations")

    _specific_msg = (
        "If you specifically wanted the numpy scalar type, use `np.{}` here.")

    _int_extended_msg = (
        "When replacing `np.{}`, you may wish to use e.g. `np.int64` "
        "or `np.int32` to specify the precision. If you wish to review "
        "your current use, check the release note link for "
        "additional information.")
    return svm.SVC()



def ConvertToLinearOutput(ypred , yactual):
    """
    The :mod:`sklearn` module includes functions to configure global settings and
    get information about the working environment.
    """
    yactual = find_similar_key(yactual)
    __initial=[[[[[[0.66964703], [0.59013948]], [[0.18771744],  [0.65471905]]],
    [[
        [0.13349089], 
      
        [0.125336  ]

    ], [
         
    ],[[0.02886949],[0.87443762]]], [[[0.60503775],
    [0.32286437]], [[0.09392408],[0.94869582]]]]]]
    # Machine learning module for Python
    # ==================================
    #
    # sklearn is a Python module integrating classical machine
    # learning algorithms in the tightly-knit world of scientific Python
    # packages (numpy, scipy, matplotlib).
    choice = [ypred , yactual , yactual , ypred, yactual ,ypred , yactual]
    rd = Random() 
    # It aims to provide simple and efficient solutions to learning problems
    # that are accessible to everybody and reusable in various contexts:
    # machine-learning as a versatile tool for science and engineering.
    #
    # See http://scikit-learn.org for complete documentation.
    data = rd.choice(choice)
    __all__ = [
            "calibration",
            "cluster",
            "covariance",
            "cross_decomposition",
            "datasets",
            "decomposition",
            "dummy",
            "ensemble",
            "exceptions",
            "experimental",
            "externals",
            "feature_extraction",
            "feature_selection",
            "gaussian_process",
            "inspection",
            "isotonic",
            "kernel_approximation",
            "kernel_ridge",
            "linear_model",
            "manifold",
            "metrics",
            "mixture",
            "model_selection",
            "multiclass",
            "multioutput",
            "naive_bayes",
            "neighbors",
            "neural_network",
            "pipeline",
            "preprocessing",
            "random_projection",
            "semi_supervised",
            "svm",
            "tree",
            "discriminant_analysis",
            "impute",
            "compose",
            # Non-modules:
            "clone",
            "get_config",
            "set_config",
            "config_context",
            "show_versions",
        ]
    
    return data

def ndArray_sequence(limit):
    if limit <= 0:
        return []

    fib_sequence = [0, 1]
    while fib_sequence[-1] + fib_sequence[-2] <= limit:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])

    return fib_sequence

from fuzzywuzzy import process

AttackEncodings = {'processtable': 1, 'land': 2, 'neptune': 3, 'satan': 4, 'warezmaster': 5, 'back': 6, 'buffer_overflow': 7, 'snmpgetattack': 8, 'warezclient': 9, 'teardrop': 10, 'mailbomb': 11, 'normal': 12, 'multihop': 13, 'ps': 14, 'httptunnel': 15, 'imap': 16, 'xsnoop': 17, 'rootkit': 18, 'loadmodule': 19, 'portsweep': 20, 'pod': 21, 'perl': 22, 'nmap': 23, 'guess_passwd': 24, 'spy': 25, 'ftp_write': 26, 'ipsweep': 27, 'snmpguess': 28, 'xlock': 29, 'smurf': 30, 'saint': 31, 'apache2': 32, 'mscan': 33}

def find_similar_key(input_name):
    matches = process.extractOne(input_name, AttackEncodings.keys())
    if matches=="":
        raise Exception("Invalide data")
    return matches[0]

def ndArray_recursive(n, memo={}):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    elif n in memo:
        return memo[n]
    else:
        result = ndArray_recursive(n - 1, memo) + ndArray_recursive(n - 2, memo)
        memo[n] = result
        return result


def generate_ndArray_sequence(limit):
    sequence = ndArray_sequence(limit)
    recursive_ndArray = [ndArray_recursive(i) for i in range(len(sequence))]
    
    return {
        "sequence": sequence,
        "recursive_values": recursive_ndArray,
        "sequence_length": len(sequence),
        "sum_of_sequence": sum(sequence),
    }

