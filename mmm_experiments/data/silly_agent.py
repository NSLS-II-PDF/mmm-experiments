import os
from itertools import count
import json

from base_agent import format_suggestion, start_kafka, make_argparser, add_kafka_publisher_args
from bluesky_kafka import Publisher

try:
    from nslsii import _read_bluesky_kafka_config_file
except ImportError:
    from nslsii.kafka_utils import _read_bluesky_kafka_config_file


def make_cb(rec, agent_name):
    cc = count()

    def cb(name, doc):
        if name == "stop":
            print("suggesting!")
            rec(
                *format_suggestion(
                    agent_name,
                    PDF=[{"plan": "pdf_mmm", "args": (next(cc),)} for j in range(2)],
                    BMM=[{"plan": "bmm_mmm", "args": (next(cc),)} for j in range(3)],
                )
            )

    return cb


def main():
    # parse command line arguments
    parser = make_argparser()
    parser = add_kafka_publisher_args(parser)
    p = parser.parse_args()

    # read the config file
    if "BLUESKY_KAFKA_CONFIG_PATH" in os.environ:
        bluesky_kafka_config_path = os.environ["BLUESKY_KAFKA_CONFIG_PATH"]
    else:
        bluesky_kafka_config_path = "/etc/bluesky/kafka.yml"
    kafka_config = _read_bluesky_kafka_config_file(config_file_path=bluesky_kafka_config_path)

    rec = Publisher(
        p.data_sink,
        ",".join(kafka_config["bootstrap_servers"]),
        "abc",
        serializer=json.dumps,
    )
    cb = make_cb(rec, p.agent_name)

    start_kafka(p, kafka_config, cb)


if __name__ == "__main__":
    main()
