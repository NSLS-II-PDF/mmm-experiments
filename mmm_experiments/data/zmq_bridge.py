import argparse
import pprint

from bluesky.callbacks.zmq import RemoteDispatcher

# need a bit more flexibility than is currently in nslsii, this file is
# vendored + a bit more plumbing
from ._kafka_publish import configure_kafka_publisher


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--zmq-address",
        required=False,
        help="zmq address to listen to.",
        default="xf28id1-ca1:5578",
        type=str,
    )
    arg_parser.add_argument(
        "--zmq-prefix",
        required=False,
        help="prefix for the document stream of interest.",
        default="an",
        type=str,
    )
    arg_parser.add_argument(
        "--beamline",
        required=False,
        help="Beamline TLA where we are.  Used for format kafka topic name.",
        default="pdf",
        type=str,
    )
    arg_parser.add_argument(
        "--document-source",
        required=False,
        help="The source of documents. Used to format kafka topic name ",
        default="pdfstream",
        type=str,
    )
    args = arg_parser.parse_args()
    pprint.pprint(args)
    start(**vars(args))


def start(zmq_address, zmq_prefix, beamline, document_source):
    # make the from zmq dispatcher
    rd = RemoteDispatcher(zmq_address, prefix=zmq_prefix.encode("ascii"))
    # subscribes inside, this pushes the documets to kafka
    configure_kafka_publisher(rd, beamline, document_source=document_source)
    # start the dispatcher
    rd.start()


if __name__ == "__main__":
    main()
