#ifndef TKDNN_PYTHONAPI_H
#define TKDNN_PYTHONAPI_H

#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>
#include <malloc.h>
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "Yolo3Detection.h"
#include "utils.h"

typedef struct {
    float x, y, w, h;
} BBox;

typedef struct {
    int cl;
    BBox bbox;
    float prob;
    char name[20];
} Detection;

bool isPerson(const char *className);
PyObject *BoundingBoxToPyObject(BBox &bbox);
Detection *get_network_boxes(tk::dnn::Yolo3Detection *net, float thresh, int batch_num, int *pnum);

extern "C" {

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} Image;

tk::dnn::Yolo3Detection *load_network(char *net_cfg, int n_batch);
Image *make_images(int w, int h, int c, int batch_size);
void copy_image_from_bytes(Image im, unsigned char *pdata);
void do_inference(tk::dnn::Yolo3Detection *net, Image *images);
PyObject *get_output(tk::dnn::Yolo3Detection *net, float thresh, int batch_num);
};

#endif //TKDNN_PYTHONAPI_H
