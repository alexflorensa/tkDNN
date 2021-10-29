#ifndef TKDNN_TKDNNLIB_H
#define TKDNN_TKDNNLIB_H

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

tk::dnn::Yolo3Detection* load_network(char* net_cfg, int n_classes, int n_batch);

};
#endif //TKDNN_TKDNNLIB_H

