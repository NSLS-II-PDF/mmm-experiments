# Fantastic Agents and How to Use Them
In general, the Agents subscribe to a stream of data, and process `tell` each time relevant data 
is produced. They can `ask` or `report` continuously, or only on command. The `ask` can go directly
to the qserver, or to an Adjudicator (via Kafka).
Every step the agent takes in the tell--report--ask cycle is documented in a stream with Tiled. 
The agents listen to second Kafka topic for Human/Adjudicator directives (see below).  


## Human in the loop
There are two ways for the human to engage in the loop: (i) through and Epics interfaced Adjudicator with soft IOCs,
 (ii) by passing plans that expose methods of the agents who are listening to a Kafka topic for commands, and 
(iii) by using the JupyterHub to process and visualize the agent document stream.

### Agent directives
Each beamline should have the following plan which allows calling any method of the agent class. The agents
will be launched remotely, and note of their names should be taken from the logs.
https://github.com/NSLS-II-PDF/mmm-experiments/blob/a1b3c7efc6d57a463f003e22524a2b25340d7b1f/mmm_experiments/agents/_plans.py#L18-L51

The recommended methods to use are in the following: 

https://github.com/NSLS-II-PDF/mmm-experiments/blob/a1b3c7efc6d57a463f003e22524a2b25340d7b1f/mmm_experiments/agents/base.py#L281-L295
https://github.com/NSLS-II-PDF/mmm-experiments/blob/4a417ad8fb9102086db1f019e90eda33b2afa25e/mmm_experiments/agents/base.py#L302-L306
https://github.com/NSLS-II-PDF/mmm-experiments/blob/a1b3c7efc6d57a463f003e22524a2b25340d7b1f/mmm_experiments/agents/base.py#L362-L393
https://github.com/NSLS-II-PDF/mmm-experiments/blob/a1b3c7efc6d57a463f003e22524a2b25340d7b1f/mmm_experiments/agents/base.py#L429-L437


## Mixins
Most agents are designed to incorporate Mixin classes to provide similar functionality
to multiple beamilines. In general a beamline specific agent (`pdf.PDFAgent`) will fill in the
abstract methods and properties of the `base.Agent` that are related to communication, and
provide some other beamline specific features (like how to unpack a run, plan name, kwargs, etc) 
and guarentee properties: 
- relative_bounds
- measurement_origin


With these in place, the Mixin classes are mostly there to provide a stateful approach to:
- `tell`
- `report`
- `ask`

It is best practice to keep heavy computation in the report and ask, because these can be toggled
on and off, and triggered remotely. These three methods are wrapped up in the base class to ensuree
effective documentation of agents.  

## Joint Agents
A successful approach was built for the independence day experiment, but is WIP for election day. 


