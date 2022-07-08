import numpy
from PIL import Image


class BMM_JPEG_HANDLER:
    def __init__(self, resource_path):
        # resource_path is really a template string with a %d in it
        self._template = resource_path

    def __call__(self, index):
        filepath = self._template % index
        return numpy.asarray(Image.open(filepath))


def patch_descriptor(doc):
    # Add more specific numpy-style data type, "dtype_str", if not present.
    if "usbcam1_image" in doc["data_keys"]:
        doc["data_keys"]["usbcam1_image"]["dtype_str"] = "|u1"
    if "usbcam2_image" in doc["data_keys"]:
        doc["data_keys"]["usbcam2_image"]["dtype_str"] = "|u1"
    if "xascam_image" in doc["data_keys"]:
        doc["data_keys"]["xascam_image"]["dtype_str"] = "|u1"
    if "xrdcam_image" in doc["data_keys"]:
        doc["data_keys"]["xrdcam_image"]["dtype_str"] = "|u1"
    if "anacam_image" in doc["data_keys"]:
        doc["data_keys"]["anacam_image"]["dtype_str"] = "|u1"
    # And so on for the others...
    return doc
