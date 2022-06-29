===============
mmm-experiments
===============

.. image:: https://img.shields.io/pypi/v/mmm-experiments.svg
        :target: https://pypi.python.org/pypi/mmm-experiments


Python package for running multimodal madness experiments with BMM

* Free software: 3-clause BSD license
* Documentation: (COMING SOON!) https://NSLS-II-PDF.github.io/mmm-experiments.

Developer Instructions
----------------------
For the sanity of your friends and colleagues, please install
pre-commit too keep your code black, flaked, and imports sorted.

.. code-block::
    git clone https://github.com/NSLS-II-PDF/mmm-experiments
    cd mmm-experiments
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements-dev.txt
    pre-commit install

Proposed Package Structure
--------------------------
- **agents**
    - base
    - pdf
    - bmm
- **data**: processing and access controls for each beamline
    - utils
    - pdf
    - bmm
- **viz**: visualization methods for data and agents
    - utils
    - pdf
    - bmm
- **comms**: communication protocols and configurations
    - bmm_plans
    - pdf_plans
    - http
    - kafka

Features
--------

* TODO
