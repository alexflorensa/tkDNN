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

extern "C" {

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} Image;

typedef struct {
    float x, y, w, h;
} BBox;

typedef struct {
    int cl;
    BBox bbox;
    float prob;
    char name[20];
} Detection;

tk::dnn::Yolo3Detection *load_network(char *net_cfg, int n_batch);
PyObject *get_output(tk::dnn::Yolo3Detection *net, float thresh, int batch_num);
};
#endif //TKDNN_PYTHONAPI_H
