from ctypes import CDLL, RTLD_GLOBAL, c_int, POINTER, c_float, Structure, c_char_p, c_void_p, c_char, py_object, \
    pointer, Array
import cv2
import time


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("cl", c_int),
                ("bbox", BOX),
                ("prob", c_float),
                ("name", c_char * 20),
                ]


lib = CDLL("./build/libPythonApi.so", RTLD_GLOBAL)

load_network = lib.load_network
load_network.argtypes = [c_char_p, c_int]
load_network.restype = c_void_p

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

do_inference = lib.do_inference
do_inference.argtypes = [c_void_p, IMAGE, c_char_p]

get_output = lib.get_output
get_output.argtypes = [c_void_p, c_float, c_int]
get_output.restype = py_object


def resizePadding(image, height, width):
    desized_size = height, width
    old_size = image.shape[:2]
    max_size_idx = old_size.index(max(old_size))
    ratio = float(desized_size[max_size_idx]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    if new_size > desized_size:
        min_size_idx = old_size.index(min(old_size))
        ratio = float(desized_size[min_size_idx]) / min(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = desized_size[1] - new_size[1]
    delta_h = desized_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return image

def detect_image(net, darknet_image, frame_data, thresh=.5):
    do_inference(net, darknet_image, frame_data)
    dets = get_output(net, 0.5, 0)
    print(dets)

    return dets


def loop_detect(detect_m, video_path):
    stream = cv2.VideoCapture(video_path)
    start = time.time()
    cnt = 0
    while stream.isOpened():
        ret, image = stream.read()
        if ret is False:
            break
        # image = resizePadding(image, 512, 512)
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(frame_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        detections = detect_m.detect(image)
        cnt += 1
    end = time.time()
    print("frame:{},time:{:.3f},FPS:{:.2f}".format(cnt, end-start, cnt/(end-start)))
    stream.release()


# class myThread(threading.Thread):
#    def __init__(self, func, args):
#       threading.Thread.__init__(self)
#       self.func = func
#       self.args = args
#    def run(self):
#       # print ("Starting " + self.args[0])
#       self.func(*self.args)
#       print ("Exiting " )


class YOLO4RT(object):
    def __init__(self,
                 input_size=512,
                 weight_file='./yolo4_fp16.rt',
                 conf_thres=0.3):
        self.input_size = input_size
        self.metaMain =None
        self.model = load_network(weight_file.encode("ascii"), 1)
        self.darknet_image = make_image(input_size, input_size, 3)
        self.thresh = conf_thres

    def detect(self, image):
        try:
            frame_data = image.ctypes.data_as(c_char_p)
            detections = detect_image(self.model, self.darknet_image, frame_data, thresh=self.thresh)

            return detections
        except Exception as e_s:
            print(e_s)


if __name__ == '__main__':
    input_path = "/home/alex/Projects/jd/val-videos/korean_walking.mp4"
    detect_m = YOLO4RT(weight_file="build/yolo4_fp32.rt")
    loop_detect(detect_m, input_path)
