.. _getting_started:

###############
Getting Started
###############

Since this is a Sunday afternoon project, there is not yet a way to install the
package. An easy and dirty way is to add the package into your path for the moment::

  import sys

  path_to_package = "/path/to/ragger_duck"
  sys.path.append(path_to_package)

You can check the file `requirements.txt` for the required packages. We also provide
a `environment.yml` file for conda users. You might want to check which version of
some libraries you are installing to take profit of the GPU acceleration.

We did not commit the folder "/models" to the repository but the script `scripts/exp.py`
expect the LLM model to be in this folder.
