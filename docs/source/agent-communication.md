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

### Some Kafka caveats associated with consuming distinct topics
Kafka preserves no guarantee of ordering between two topics, and (cannot poll topics individually)[https://stackoverflow.com/questions/55798164/kafka-have-each-poll-call-only-consume-from-one-topic-at-a-time].
To avoid having to thread multiple consumers in a single agent (the process taken below for adjudicating), we must stay
aware of (race conditions)[https://www.cloudkarafka.com/blog/a-dive-into-multi-topic-subscriptions-with-apache-kafka.html].
In general, an agent will be told about all data that appears on its Kafka topics, and will pause loading/unpacking data
while executing a command like `generate_report`, since this happens in the same process.
In short summary: 
1. If your `tell` is slower than your experiment, your processing is bad. Go the route of pdfstream.
2. If your `ask` is slower than your experiment, consider having it in an open loop, 
and triggering at select times for a minor a delay in data loading. I.e. `disable_continuous_suggsting`.
3. The above goes the same for an expensive `report`, though this is less likely. 
4. If your `ask` is much slower than the experiment time, to the point that you would lose messages from the Kafka topic
queue, it should be a fully external process. In any case, the agent must bne very clever to justify this,
or an experiment must be so expensive that it is justifiable. 


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
suggestions for more than one or more beamlines. Here shown as dictionaries, it
is implemented using pydantic models which support the nested serialization needed for kafka:

```python
name, doc = [
    "agent_suggestions",
    {
        # name of agent, used by adjudicator
        "agent": "aardvark",
        # unique ID of this ask / batch of suggestions
        "uid": "086ff562-c997-48c7-9fbb-9cbe3c362ff0",
        # per-beamline suggestions
        "suggestions": {
            "pdf": [
                {
                    # unique id of this specific suggestion from the `ask` call
                    "ask_uid": "4dc2c5f2-843b-47e1-aa56-d5bf6d682bff",
                    # the plan + args + kwargs requested
                   "plan_name": "pdf_mmm",
                   "plan_args": [10],
                   "plan_kwargs": {}
                },
                {
                    "ask_uid": "d4f6eae5-7789-48f7-82c1-ce51b9118c8b",
                    "plan_name": "pdf_mmm",
                    "plan_args": [11],
                    "plan_kwargs": {}
                },
            ],
            "BMM": [
                {
                    "ask_uid": "5e0826b8-8af9-403b-8756-52ff1de614c5",
                    "plan_name": "bmm_mmm",
                    "plan_args": [12],
                    "plan_kwargs": {}
                },
                {
                    "ask_uid": "9f81ab42-75d2-42ab-b532-329977bcf33c",
                    "plan_name": "bmm_mmm",
                    "plan_args": [13],
                    "plan_kwargs": {}
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
