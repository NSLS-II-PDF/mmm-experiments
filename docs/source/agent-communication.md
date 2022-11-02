# A Rundown on How the Agents Communicate

## Isolated Agents
Each agent listens to and consumes a Kafka topic with the relevant data from the beamline.
The agents also consume data from a Kafka topic that will trigger agent actions by calling agent methods
with documents. 

Every new agent must implement `tell`--`report`--`ask`. 
At each stage of this cycle the agent wraps these actions and writes data to MongoDB via Tiled. 
When the agent hears about new data from its beamline, it updates itself via a `tell` method, that unpacks the relevant data from Tiled.
When appropriate, the agent can generate a report using the `report` method, which can be triggered *via* the `generate_report` method.
Lastly, `ask` is wrapped up in some queue management and Tiled writing by the `generate_report` method. 

### Recommended methods to expose to beamline scientists
Methods should be packaged into documents/dictionaries with these specific keys: `{"action":<method_name>, "args":..., "kwargs":...}`.

- enable_continuous_reporting
- disable_continuous_reporting
- enable_continous_suggesting
- disable_continuous_suggesting
- restart
- generate_report
- add_suggestions
- tell_agent_by_uid

A bluesky plan to send documents to the appropriate kafka node is present in  mmm_experiments/agents/_plans.py. 

## Multiple Agents Adjudicated by a Third Party
