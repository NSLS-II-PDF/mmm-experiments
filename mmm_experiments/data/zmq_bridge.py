import argparse
from copy import deepcopy
import pprint
from tiled.client import from_profile
import datetime

from bluesky.callbacks.zmq import RemoteDispatcher

from event_model import RunRouter
from suitcase.msgpack import Serializer
from zmq import PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_MESSAGE

# need a bit more flexibility than is currently in nslsii, this file is
# vendored + a bit more plumbing
from ._kafka_publisher import configure_kafka_publisher


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
    arg_parser.add_argument(
        "--msgpack-path",
        required=False,
        help="A direcotry to dump a msgpack log of the documents seen to.",
        default="",
        type=str,
    )
    args = arg_parser.parse_args()
    pprint.pprint(args)
    start(**vars(args))


def transform_factory(striped_keys):
    def strip_dict(inp):
        return {k: v for k, v in inp.items() if k not in striped_keys}

    def strip_keys(name, doc):
        if name in {"start", "stop"}:
            pass
        elif name == "descriptor":
            doc = deepcopy(doc)
            doc["data_keys"] = strip_dict(doc["data_keys"])
            for prefix in ["chi", "iq", "sq", "fq", "gr"]:
                for k, v in doc["data_keys"].items():
                    if not k.startswith(prefix):
                        continue
                    if len(v["shape"]) == 0:
                        continue
                    v["dims"] = (f"{prefix}_index",)
            for k, v in doc["object_keys"].items():
                doc["object_keys"][k] = [_ for _ in v if _ not in striped_keys]

        elif name == "event" or name == "event_page":
            doc = deepcopy(doc)
            for entry in ["data", "timestamps"]:
                doc[entry] = strip_dict(doc[entry])

        else:

            raise NotImplementedError(f"no support for {name}")

        return name, doc

    return strip_keys


def start(zmq_address, zmq_prefix, beamline, document_source, msgpack_path):

    # make the from zmq dispatcher
    rd = RemoteDispatcher(zmq_address, prefix=zmq_prefix.encode("ascii"))
    if msgpack_path:
        dump_rr = RunRouter(factories=[lambda name, doc: ([Serializer(msgpack_path)], [])])
        rd.subscribe(dump_rr)
    rd.subscribe(
        lambda name, doc: print(name, datetime.datetime.fromtimestamp(doc["time"]).isoformat())
        if name in {"start", "stop"}
        else None
    )

    transformer = transform_factory({"dk_sub_image", "mask"})
    # subscribes inside, this pushes the documets to kafka
    configure_kafka_publisher(rd, beamline, document_source=document_source, transformer=transformer)
    # get the document-aware sandbox
    cat = from_profile("pdf_bluesky_sandbox")
    # subscribe to insert the reduced data as well
    rd.subscribe(lambda name, doc: cat.v1.insert(*transformer(name, doc)))
    # start the dispatcher
    rd.start()


if __name__ == "__main__":
    main()
