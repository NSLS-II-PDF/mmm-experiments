import datetime
import inspect
import pprint
import uuid
import warnings

import nslsii
from bluesky_kafka import BlueskyConsumer, RemoteDispatcher


def print_message(name, doc):
    """Convenience print"""
    print(f"{datetime.datetime.now().isoformat()} document: {name}\n" f"contents: {pprint.pformat(doc)}\n")


def process_kafka_messages(beamline_acronym, data_source="runengine", target=None):
    """
    Parameters
    ----------
    beamline_acronym : str
        the lowercase TLA

    data_source : str, optional
        The source of the documents on the beamline (e.g. 'runengine' or 'pdfstream')

    target : Callable[(str, Document), None], optional
        Function to pass the documents from kafka to. If not specified, defaults to printing
        the documents
    """

    if target is None:
        target = print_message

    kafka_config = nslsii._read_bluesky_kafka_config_file(config_file_path="/etc/bluesky/kafka.yml")

    # this consumer should not be in a group with other consumers
    #   so generate a unique consumer group id for it
    unique_group_id = f"echo-{beamline_acronym}-{data_source}-{str(uuid.uuid4())[:8]}"

    kafka_dispatcher = RemoteDispatcher(
        topics=[f"{beamline_acronym}.bluesky.{data_source}.documents"],
        bootstrap_servers=",".join(kafka_config["bootstrap_servers"]),
        group_id=unique_group_id,
        consumer_config=kafka_config["runengine_producer_config"],
    )

    kafka_dispatcher.subscribe(target)
    kafka_dispatcher.start()


class AgentConsumer(BlueskyConsumer):
    """Specific consumer for mmm agents. Consumes both bluesky docs and mmm docs"""

    def __init__(self, *args, agent, beamline_tla, bluesky_callbacks=(), **kwargs):
        """

        Parameters
        ----------
        args :
            Positional args for BlueskyConsumer.
            topics, bootstrap_servers, group_id
        agent : mmm_experiments.agents.base.Agent
            Instance of agent to use for triggering agent actions
        beamline_tla : str
            Beamline three letter acronym, used for setup and assurance
        bluesky_callbacks : Iterable[Callable]
            List of callbacks to perform on documents. This list should be similar to callbacks used in
            RE.subscribe(), and will be processed in order.
        kwargs :
            kwargs for BlueskyConsumer
        """
        super().__init__(*args, **kwargs)
        self.tla = beamline_tla
        self.bluesky_callbacks = bluesky_callbacks
        self.agent = agent

    def process(self, msg) -> bool:
        name, doc = self._deserializer(msg.value())
        continue_polling = self.process_document(msg.topic(), name, doc)
        return continue_polling

    def process_document(self, topic, name, doc):
        keywords = topic.split(".")
        print(topic, name, doc, keywords)
        if self.tla not in keywords:
            return True
        if ("mmm" in keywords and "agents" in keywords) and name == self.agent.agent_uid:

            action = doc["action"]
            args = doc["args"]
            kwargs = doc["kwargs"]
            try:
                getattr(self.agent, action)(*args, **kwargs)
            except AttributeError as e:
                warnings.warn(
                    f"Unavailable action sent to agent {self.agent.agent_uid} on topic: {topic}\n" f"{e}"
                )
            except TypeError as e:
                warnings.warn(
                    f"Type error for {action} sent to agent {self.agent.agent_uid} on topic: {topic}\n"
                    f"Are you sure your args and kwargs were appropriate?\n"
                    f"Args received: {args}\n"
                    f"Kwargs received: {kwargs}\n"
                    f"Expected signature: {inspect.signature(getattr(self.agent, action))}\n"
                    f"{e}"
                )
            return True

        else:
            for callback in self.bluesky_callbacks:
                callback(name, doc)
            return True
