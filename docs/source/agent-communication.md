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

A bluesky plan to send documents to the appropriate kafka node is present in  mmm_experiments/agents/_plans.py.

## Multiple Agents Adjudicated by a Third Party

If multiple agents are is use, or one agent is continuously making many
suggestions, it is useful to buffer all of the suggestions between a single
"adjudication" agent.  The exact heuristics in this agent will vary from a
simple knob of "always listen to agent X" to automatically selecting the best
agent from a pool of comparable agents.

In this model, all of the agents listen to kafka to be notified when new data
(raw or processed) is available and do what ever they need to run their `tell`
to update their internal state.  As part of the `generate_report` process they
will publish to a kafka topic a message with their current wish-list of
measurements.  These messages can then be logged for later consideration and
aggregation by the adjudicator process.

### Payload Details

Matching all of our kafka topics the pattern is `(name, doc)`.  In this case
the doc is a nested structure to support a single agent one or more
suggestions for more than one or more beamlines:

```python
name, doc = [
    "agent_suggestions",
    {
        # name of agent, used by adjudicator
        "agent": "aardvark",
        # unique ID of this ask / batch of suggestions
        "publish_uid": "086ff562-c997-48c7-9fbb-9cbe3c362ff0",
        # per-beamline suggestions
        "suggestions": {
            "PDF": [
                {
                    # unique id of this specific suggestion
                    "suggestion_uid": "4dc2c5f2-843b-47e1-aa56-d5bf6d682bff",
                    # the plan + args + kwargs requested
                    "request": {"plan": "pdf_mmm", "args": [10], "kwargs": {}},
                },
                {
                    "suggestion_uid": "d4f6eae5-7789-48f7-82c1-ce51b9118c8b",
                    "request": {"plan": "pdf_mmm", "args": [11], "kwargs": {}},
                },
            ],
            "BMM": [
                {
                    "suggestion_uid": "5e0826b8-8af9-403b-8756-52ff1de614c5",
                    "request": {"plan": "bmm_mmm", "args": [12], "kwargs": {}},
                },
                {
                    "suggestion_uid": "9f81ab42-75d2-42ab-b532-329977bcf33c",
                    "request": {"plan": "bmm_mmm", "args": [13], "kwargs": {}},
                },
                {
                    "suggestion_uid": "f639d2b0-d032-422e-bfc7-4fdad3b2d77f",
                    "request": {"plan": "bmm_mmm", "args": [14], "kwargs": {}},
                },
            ],
        },
    },
]

```

The outer and inner layers may contain additional information, all keys in
`"suggestions"` are assumed to be beamline TLAs.

The outer `publish_id` is a unique id for the ask from the agent.  The
per-suggestion aids are so that the adjudicator can track if it has added that
particular suggestion to the queue before.

All of `"agent_name"`, `"publish_uid"`, and `"suggestion_uid"` should be added
to the metadata when the plan is added to the queue by the adjudicator.


### Example use cases

1. Bayesian algorithms may need significant amount of initial data so run a
   grid scan agent (at ever increasing grid size) until the Bayes / Gaussian
   process model starts to make good suggestions and then switch to it.
2. A suite of NMF models with different numbers of components may be run.  The
   one with the best looking loss function is selected to be listened to.
3. If a model is much slower to compute than it is to take data, it may provide
   a long list of requested measurements.  When a new set is available
   suggestions should be taken from the new list and the old list discard.
4. If a higher-priority agent is out of suggestions, fallback to the next agent is line.


### Implementation
