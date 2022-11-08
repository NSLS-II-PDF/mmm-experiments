from pathlib import Path
import os
import logging
from collections import namedtuple

try:
    from nslsii import _read_bluesky_kafka_config_file
except ImportError:
    from nslsii.kafka_utils import _read_bluesky_kafka_config_file

logger = logging.getLogger(name="mmm.kafka")

"""
A namedtuple for holding details of the publisher created by
_subscribe_kafka_publisher.
"""
_SubscribeKafkaPublisherDetails = namedtuple(
    "SubscribeKafkaPublisherDetails",
    {"beamline_topic", "bootstrap_servers", "producer_config", "re_subscribe_token"},
)


def _subscribe_kafka_publisher(
    RE,
    beamline_name,
    bootstrap_servers,
    producer_config,
    _publisher_factory=None,
    *,
    document_source="runengine",
    filter=lambda doc: True,
    transformer=lambda name, doc: (name, doc),
):
    """
    Subscribe a RunRouter to the specified RE to create Kafka Publishers.
    Each Publisher will publish documents from a single run to the
    Kafka topic "<beamline_name>.bluesky.<document_source>.documents".

    Parameters
    ----------
    RE : RunEngine
        the RunEngine to which the RunRouter will be subscribed
    beamline_name : str
        beamline start_name, for example "csx", to be used in building the
        Kafka topic to which messages will be published
    bootstrap_servers : str
        Comma-delimited list of Kafka server addresses as a string such as ``'10.0.137.8:9092'``
    producer_config : dict
        dictionary of Kafka Producer configuration settings
    _publisher_factory : callable, optional
        intended only for testing, default is bluesky_kafka.Publisher, optionally specify a callable
        that constructs a Publisher-like object
    document_source : str, optional
        The document source.
    filter : Callable[Dict[str, Any], bool]
        Return True if this run should be handled, False if not not.
    Returns
    -------
    topic: str
        the Kafka topic on which bluesky documents will be published
    runrouter_token: int
        subscription token corresponding to the RunRouter subscribed to the RunEngine
        by this function
    """
    from bluesky_kafka import Publisher
    from bluesky_kafka.utils import list_topics
    from event_model import RunRouter

    topic = f"{beamline_name.lower()}.bluesky.{document_source}.documents"

    if _publisher_factory is None:
        _publisher_factory = Publisher

    def kafka_publisher_factory(start_name, start_doc):
        if not filter(start_doc):
            return [], []
        # create a Kafka Publisher for a single run
        #   in response to a start document
        kafka_publisher = _publisher_factory(
            topic=topic,
            bootstrap_servers=bootstrap_servers,
            key=start_doc["uid"],
            producer_config=producer_config,
            flush_on_stop_doc=True,
        )

        def publish_or_abort_run(name_in, doc_in):
            """
            Exceptions _should_ interrupt the current run.
            """
            name_, doc_ = transformer(name_in, doc_in)
            try:
                kafka_publisher(name_, doc_)
            except (BaseException, Exception) as exc_:
                # log the exception and re-raise it to abort the current run
                logger = logging.getLogger("nslsii")
                logger.exception(
                    "an error occurred when %s published\n  start_name: %s\n  doc %s",
                    kafka_publisher,
                    name_,
                    doc_,
                )
                raise exc_

        try:
            # on each start document call list_topics to test if we can connect to a Kafka broker
            cluster_metadata = list_topics(
                bootstrap_servers=bootstrap_servers,
                producer_config=producer_config,
                timeout=5.0,
            )
            logging.getLogger("nslsii").info("connected to Kafka broker(s): %s", cluster_metadata)
            return [publish_or_abort_run], []
        except (BaseException, Exception) as exc:
            # log the exception and re-raise it to indicate no connection could be made to a Kafka broker
            nslsii_logger = logging.getLogger("nslsii")
            nslsii_logger.exception("'%s' failed to connect to Kafka", kafka_publisher)
            raise exc

    rr = RunRouter(factories=[kafka_publisher_factory])
    runrouter_token = RE.subscribe(rr)

    # log this only once
    logging.getLogger("nslsii").info("RE will publish documents to Kafka topic '%s'", topic)

    subscribe_kafka_publisher_details = _SubscribeKafkaPublisherDetails(
        beamline_topic=topic,
        bootstrap_servers=bootstrap_servers,
        producer_config=producer_config,
        re_subscribe_token=runrouter_token,
    )

    return subscribe_kafka_publisher_details


"""
A namedtuple for holding details of the publisher created by
_subscribe_kafka_queue_thread_publisher.
"""
_SubscribeKafkaQueueThreadPublisherDetails = namedtuple(
    "SubscribeKafkaQueueThreadPublisherDetails",
    {
        "beamline_topic",
        "bootstrap_servers",
        "producer_config",
        "publisher_queue_thread_details",
        "re_subscribe_token",
    },
)


def configure_kafka_publisher(RE, beamline_name, override_config_path=None, **kwargs):
    """Read a Kafka configuration file and subscribe a Kafka publisher to the RunEngine.

    A configuration file is required. Environment variable BLUESKY_KAFKA_CONFIG_FILE
    will be checked if `configuration_file_path` is not specified. Otherwise the default
    path `/etc/bluesky/kafka.yml` will be read.

    The intention is that the default path is used in production. The environment variable
    allows for modifying a deployed system and the parameter is useful for testing.

    See `tests/test_kafka_configuration.py` for an example configuration file.
    """
    bluesky_kafka_config_path = None
    kafka_publisher_details = None

    if override_config_path is not None:
        bluesky_kafka_config_path = override_config_path
    elif "BLUESKY_KAFKA_CONFIG_PATH" in os.environ:
        bluesky_kafka_config_path = os.environ["BLUESKY_KAFKA_CONFIG_PATH"]
    else:
        bluesky_kafka_config_path = "/etc/bluesky/kafka.yml"

    bluesky_kafka_configuration = _read_bluesky_kafka_config_file(bluesky_kafka_config_path)
    # convert the list of bootstrap servers into a comma-delimited string
    #   which is the format required by the confluent python api
    bootstrap_servers = ",".join(bluesky_kafka_configuration["bootstrap_servers"])

    kafka_publisher_details = _subscribe_kafka_publisher(
        RE,
        beamline_name=beamline_name,
        bootstrap_servers=bootstrap_servers,
        producer_config=bluesky_kafka_configuration["runengine_producer_config"],
        **kwargs,
    )

    return bluesky_kafka_configuration, kafka_publisher_details
