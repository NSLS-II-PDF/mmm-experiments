import datetime
import pprint
import uuid
from bluesky_kafka import RemoteDispatcher
import nslsii


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

    def print_message(name, doc):
        print(f"{datetime.datetime.now().isoformat()} document: {name}\n" f"contents: {pprint.pformat(doc)}\n")

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
