import argparse
import datetime
import os
import pprint
import uuid

from bluesky_kafka import RemoteDispatcher

try:
    from nslsii import _read_bluesky_kafka_config_file
except ImportError:
    from nslsii.kafka_utils import _read_bluesky_kafka_config_file


def make_argparser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--beamline",
        required=False,
        help="Beamline TLA where we are.  Used for format kafka topic name and client group name.",
        default="pdf",
        type=str,
    )
    arg_parser.add_argument(
        "--document-source",
        required=False,
        help="The source of documents. Used to format kafka topic name and client group name.",
        default="bluesky",
        type=str,
    )
    arg_parser.add_argument(
        "--group-base-name",
        required=False,
        help="The base name of the client group.  If not passed a uuid is used.",
        default=None,
        type=str,
    )
    arg_parser.add_argument(
        "--auto-offset-reset",
        required=False,
        help="How to reset the topic offset when attaching.",
        default="smallest",
        choices=["smallest", "latest", "none"],
    )

    return arg_parser


def add_kafka_publisher_args(arg_parser):
    arg_parser.add_argument(
        "--data-sink",
        required=True,
        help="name of the target topic to publish to.",
        type=str,
    )
    return arg_parser


def start_kafka(parameters, *targets):
    # handle default target
    def print_message(name, doc):
        print(f"{datetime.datetime.now().isoformat()} document: {name}\n" f"contents: {pprint.pformat(doc)}\n")

    if not len(targets):
        targets = [print_message]

    # unpack parameters to local namespace
    beamline_acronym = parameters.beamline
    document_source = parameters.document_source
    auto_offset_reset = parameters.auto_offset_reset
    group_base = parameters.group_base_name
    if group_base is None:
        group_base = str(uuid.uuid4())

    # read the config file
    if "BLUESKY_KAFKA_CONFIG_PATH" in os.environ:
        bluesky_kafka_config_path = os.environ["BLUESKY_KAFKA_CONFIG_PATH"]
    else:
        bluesky_kafka_config_path = "/etc/bluesky/kafka.yml"

    kafka_config = _read_bluesky_kafka_config_file(config_file_path=bluesky_kafka_config_path)

    # this consumer should not be in a group with other consumers
    #   so generate a unique consumer group id for it
    group_id = f"{group_base}-{beamline_acronym}-{document_source}"
    cconfig = kafka_config["runengine_producer_config"]
    cconfig["auto.offset.reset"] = auto_offset_reset
    kafka_dispatcher = RemoteDispatcher(
        topics=[f"{beamline_acronym}.{document_source}.documents"],
        bootstrap_servers=",".join(kafka_config["bootstrap_servers"]),
        group_id=group_id,
        consumer_config=cconfig,
    )

    for target in targets:
        kafka_dispatcher.subscribe(target)

    kafka_dispatcher.start()
