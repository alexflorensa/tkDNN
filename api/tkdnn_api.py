from typing import Dict, List, TYPE_CHECKING, Tuple

from ctypes import CDLL, POINTER, RTLD_GLOBAL, Structure, c_char_p, c_float, c_int, c_void_p, py_object

if TYPE_CHECKING:
    import numpy as np

Detection = Tuple[Tuple[int, ...], float]

TKDNN_LIB_PATH = "libPythonApi.so"  # add to LD_LIBRARY_PATH env var or set the path of the file.

lib = CDLL(TKDNN_LIB_PATH, RTLD_GLOBAL)


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


_load_network = lib.load_network
_load_network.argtypes = [c_char_p, c_int]
_load_network.restype = c_void_p

_make_images = lib.make_images
_make_images.argtypes = [c_int, c_int, c_int, c_int]
_make_images.restype = POINTER(IMAGE)

_copy_image_from_bytes = lib.copy_image_from_bytes
_copy_image_from_bytes.argtypes = [IMAGE, c_char_p]

_do_inference = lib.do_inference
_do_inference.argtypes = [c_void_p, POINTER(IMAGE)]

_get_output = lib.get_output
_get_output.argtypes = [c_void_p, c_float, c_int, c_int, c_int]
_get_output.restype = py_object


def load_network(model_path: str, n_batches: int) -> c_void_p:
    """
    Loads the network given a path.

    :param model_path: TensorRT Engine path
    :param n_batches: Batch size to be used on detections
    :return: Pointer to the network class
    """
    return _load_network(model_path.encode('ascii'), c_int(n_batches))


def make_images(w: int, h: int, c: int, batch_size: int) -> POINTER(IMAGE):
    """
    Allocates memory for the network input image

    :param w: Network image width.
    :param h: Network image height.
    :param c: Network image channels.
    :param batch_size: Network batch size
    :return: An structure with a pointer to the allocated memory.
    """
    return _make_images(w, h, c, batch_size)


def do_inference(network: c_void_p, tkdnn_images: POINTER(IMAGE), images: List['np.ndarray'], n_batch: int) -> None:
    """
    Does the forward pass and stores the output.

    :param network: Pointer to the network class
    :param tkdnn_images: List of Structures with a pointer to the allocated memory using make_image function.
    :param images: List of Numpy arrays that is the input images with the same dimension as the network input shape.
    """
    for i in range(n_batch):
        _copy_image_from_bytes(tkdnn_images[i], images[i].ctypes.data_as(c_char_p))
    _do_inference(network, tkdnn_images)


def get_output(
        network: c_void_p,
        threshold: float,
        batch: int,
        frame_shape: Tuple[int, int]
) -> Dict[str, List['Detection']]:
    """

    :param network: Pointer to the network class
    :param threshold: Confidence threshold of the detections
    :param batch: Which batch to get output from.
    :param frame_shape: Frame shape where detections where done.
    :return: Dictionary with class name as key and a list of tuples (bounding box, confidence) as value.
    """
    return _get_output(network, threshold, batch, *frame_shape)
