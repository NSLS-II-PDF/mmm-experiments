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

.. code-block:: bash

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

Running with a local mongo
--------------------------
- Following instructions here: https://www.mongodb.com/compatibility/docker
- `data/testing_config.yml` goes in `~/.config/tiled/profiles` or `./venv/etc/tiled/profiles`

In terminal:

.. code-block:: bash

    docker run --name mongodb -d -v /tmp/databroker:/data/db -p 27017:27017 mongo


Checking in python:

.. code-block:: python

    from tiled.client import from_profile
    from tiled.profiles import list_profiles
    from event_model import compose_run
    list_profiles()
    catalog = from_profile("testing_sandbox")
    run_bundle = compose_run(uid=None, time=None, metadata=None)
    catalog.v1.insert("start", run_bundle.start_doc)
    uid = run_bundle.start_doc["uid"]
    catalog["uid"]
    # Should output something like <BlueskyRun set() scan_id=UNSET uid='6201900e' 2022-06-30 12:49>


With mongo db live in docker, the `data/example_data.py` script will show how to write into the
database with some dummy data.
