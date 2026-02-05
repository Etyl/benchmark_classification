
   $ conda create -n benchopt-clf python=3.11 -y
   $ conda activate benchopt-clf

Install benchopt and the benchmark dependencies:

.. code-block::

   $ python -m pip install benchopt
   $ benchopt install .


**Install without conda**

Create a virtual environment, and install the requirements:

.. code-block::

   $ python3 -m venv benchopt-clf
   $ source benchopt-clf/bin/activate
   $ python -m pip install --upgrade pip
   $ python -m pip install -r requirements.txt


Run the benchmark
-----------------

.. code-block::

   $ benchopt run .

Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

