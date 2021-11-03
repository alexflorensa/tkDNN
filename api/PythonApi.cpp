#include "PythonApi.h"


Image *make_images(int w, int h, int c, int batch_size) {
    Image *out = (Image *) xcalloc(batch_size, sizeof(Image));
    for (int i = 0; i < batch_size; i++) {
        out[i].w = w;
        out[i].h = h;
        out[i].c = c;
        out[i].data = (float *) xcalloc(h * w * c, sizeof(float));
    }
    return out;
}

void copy_image_from_bytes(Image im, unsigned char *pdata) {
    memcpy(im.data, pdata, im.h * im.w * im.c);
}


tk::dnn::Yolo3Detection *load_network(char *net_cfg, int n_batch) {
    std::string net = std::string(net_cfg);
    tk::dnn::Yolo3Detection *detNN = new tk::dnn::Yolo3Detection;
    detNN->init(net, 80, n_batch, 0.3);
    return detNN;
}

void do_inference(tk::dnn::Yolo3Detection *net, Image *images) {
    std::vector<cv::Mat> batch_dnn_input;
    for (int i = 0; i < net->nBatches; i++) {
        cv::Mat frame(images[i].h, images[i].w, CV_8UC3, (unsigned char *) images[i].data);
        batch_dnn_input.push_back(frame);
    }
    net->update(batch_dnn_input, net->nBatches);

}

Detection *get_network_boxes(tk::dnn::Yolo3Detection *net, float thresh, int batch_num, int *pnum) {
    std::vector<std::vector<tk::dnn::box>> batchDetected;
    batchDetected = net->get_batch_detected();
    int nboxes = 0;
    std::vector<std::string> classesName = net->get_classesName();
    Detection *dets = (Detection *) xcalloc(batchDetected[batch_num].size(), sizeof(Detection));

    for (auto &det: batchDetected[batch_num]) {
        if (det.prob > thresh && isPerson(classesName[det.cl].c_str())) {
            dets[nboxes].cl = det.cl;
            strcpy(dets[nboxes].name, classesName[det.cl].c_str());
            dets[nboxes].bbox.x = det.x;
            dets[nboxes].bbox.y = det.y;
            dets[nboxes].bbox.w = det.w;
            dets[nboxes].bbox.h = det.h;
            dets[nboxes].prob = det.prob;
            nboxes += 1;
        }
    }
    if (pnum) *pnum = nboxes;
    return dets;
}


bool isPerson(const char *className) {
    return strcmp(className, "person") == 0;
}


PyObject *BoundingBoxToPyObject(BBox &bbox) {
    PyObject *pyBoundingBox = PyTuple_New(4);
    PyTuple_SetItem(pyBoundingBox, 0, PyFloat_FromDouble(bbox.x));
    PyTuple_SetItem(pyBoundingBox, 1, PyFloat_FromDouble(bbox.y));
    PyTuple_SetItem(pyBoundingBox, 2, PyFloat_FromDouble(bbox.w));
    PyTuple_SetItem(pyBoundingBox, 3, PyFloat_FromDouble(bbox.h));
    return pyBoundingBox;
}

PyObject *get_output(tk::dnn::Yolo3Detection *net, float thresh, int batch_num) {
    int pnum;
    Detection *dets = get_network_boxes(net, thresh, batch_num, &pnum);
    PyGILState_STATE gilState = PyGILState_Ensure();
    PyObject *finalDets = PyDict_New();
    for (int i = 0; i < pnum; i++) {
        Detection &det = dets[i];
        PyObject *key = PyUnicode_FromString(det.name);
        PyObject *detList;
        if (!PyDict_Contains(finalDets, key)) {
            detList = PyList_New(0);
            PyDict_SetItem(finalDets, key, detList);
        } else { detList = PyDict_GetItem(finalDets, key); }

        PyObject *detection = PyTuple_New(2);
        PyTuple_SetItem(detection, 0, BoundingBoxToPyObject(det.bbox));
        PyTuple_SetItem(detection, 1, PyFloat_FromDouble(det.prob));
        PyList_Append(detList, detection);
    }
    PyGILState_Release(gilState);
    return finalDets;
}