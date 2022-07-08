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


Setting up Tiled Access
-----------------------
- `data/local_mmm_config.yml` goes in `~/.config/tiled/profiles` or `./venv/etc/tiled/profiles`
- `data/bmm_patches.py` goes in `~/.config/tiled/` or `./venv/etc/tiled/`
- `data/make_tiled_links.sh` goes in `~/.config/tiled/` or `./venv/etc/tiled/`
- Execute make_tiled_links.sh from the tiled directory to create a set of soft links.
Checking with python:

.. code-block:: python

    from tiled.client import from_profile
    from tiled.profiles import list_profiles
    list_profiles()
    catalog = from_profile("bmm")
    catalog = from_profile("bmm_bluesky_sandbox")
    catalog = from_profile("pdf_bluesky_sandbox")

Running with a local mongo
--------------------------
- Following instructions here: https://www.mongodb.com/compatibility/docker
- `data/testing_config.yml` goes in `~/.config/tiled/profiles` or `./venv/etc/tiled/profiles`
    - This file goes to similar locations on the remote machine running the agents on the science network.

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


Qserver Notes
-------------
On srv1 to launch and perform simple work. The RE manager is launched by systemd.

.. code-block:: bash

    conda activate $BS_ENV
    qserver environment open
    qserver status
    qserver queue add plan '{"name": "mv", "args":["xafs_x", 50, "xafs_y", 125]}' # Dumb plan, check numbers
    qserver queue start
    qserver environment close


Some example tests using the API are shown here:
https://gist.github.com/dmgav/87dc6c2f7b0bb5775afb5e1277176850


=================
Adding a new plan
=================

In :code:`/nsls2/data/TLA/shared/config/bluesky/profile_collection/startup`, adjust :code:`user_group_permissions.yaml`
to include :code:`':^agent_'` in user_groups:root:allowed_plans.

`qserver permissions get` should show this.

Adding a plan to a start up file (like :code:`def agent_test_plan()` in :code:`99-agent_plans.py`),
then closing the qserver environment, and updating :code:`existing_plans_and_devices.yaml` using  the CLI
will make the plan available. The following commands as the operator account should update the accesible plans.

.. code-block:: bash

    cd ~/.ipython/profile_qs/startup
    unset SUDO_USER # A quirk of BMM's dependence on larch
    qserver environment close
    qserver status
    qserver-list-plans-devices --startup-dir . # updates existing_plans_and_devices.yaml
    # Check exiting plans
    qserver existing plans
    qserver environment open
    qserver status
    # waiting for  'worker_environment_exists': True, 'worker_environment_state': 'idle'}
    # The following line is sometimes necessary...
    qserver permissions reload
    # Check the new plan is allowed
    qserver allowed plans



Dealing with PDF Analyzed data
------------------------------

To start the zmq -> kafka / mongo bridge
:code: `python -m mmm_experiments.data.zmq_bridge`
in an env where the package is installed.
This will publish to the topic pdf.bluesky.pdfstream.documents and insert into the pdf_bluesky_sandbox databroker.
To work this strips out the two images from the pdfstream data stream.

