.. image:: https://raw.githubusercontent.com/snurr-group/pyRAST/refs/heads/main/docs/source/_static/pyrast_logo_200x200.png
    :alt: pyRAST logo
    :align: center
    :width: 200px

========
Overview
========

Python Real Adsorbed Solution Theory (pyRAST) is a package for modeling adsorption
using ideal and real adsorbed solution theories. pyRAST was designed to be user-friendly
and flexible, allowing users to easily control the fitting and calculation process.


.. badges

|status| |docs| |license|


.. |status| image:: https://www.repostatus.org/badges/latest/active.svg
    :alt: Project Status: Active - The project has reached a stable, usable state and is being actively developed.
    :target: https://www.repostatus.org/#active

.. |docs| image:: https://readthedocs.org/projects/pyrast/badge/?style=flat
    :target: https://readthedocs.org/projects/pyrast
    :alt: Documentation Status

.. |license| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: Project License

Features
--------
- Analytical single component isotherm fitting
- Vacancy solution theory (VST) isotherm fitting
- Single component interpolator isotherms
- Activity coefficient fitting for binary adsorbed mixtures (component loading or total loading)
- Multicomponent forward and reverse IAST calculations
- Binary forward and reverse RAST calculations

Usage
-----
Please see the `tutorial <https://pyrast.readthedocs.io/en/latest/tutorial/index.html>`__ for a detailed overview
of pyRAST's functionality and usage. If you are looking for detailed documentation, please see the
`reference <https://pyrast.readthedocs.io/en/latest/reference/index.html>`__ section. For a detailed
description of the underlying theory, please see the manuscript (link to be added).

Installation
============
pyRAST is available on pyPI and can be installed using pip: ::

    pip install pyrast


Citation
========
If you used pyRAST in your research, please cite the following paper: ::

    To be added

pyRAST was built on the foundation of `pyIAST <https://github.com/CorySimon/pyIAST>`__ by Cory Simon
and pulled ideas from `pyGAPS <https://github.com/pauliacomi/pyGAPS>`__ by Paul Iacomi. Please
check out the work of these developers, too!

Acknowledgements
================
pyRAST was developed by Jonah Finkelstein in the Snurr Research Group at Northwestern University.
The development of pyRAST was supported by: TBD

Development
===========
If you wish to install pyRAST in development mode, clone the repository and run the following command in the root directory: ::

    pip install -e .[dev]

If you wish to contribute to the development of pyRAST, please submit a `pull request <https://github.com/snurr-group/pyRAST/pulls>`__ 
on the GitHub Repository or contact Jonah Finkelstein.

Questions?
==========
If you have any questions, please contact Jonah Finkelstein at jonahfinkelstein2030@u.northwestern.edu.
Alternatively, you can open an `issue <https://github.com/snurr-group/pyRAST/issues>`__ on the GitHub repository.
