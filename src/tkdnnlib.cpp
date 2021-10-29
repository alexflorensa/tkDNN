#include "../include/tkdnnlib.h"

extern "C"
{


void copy_image_from_bytes(Image im, unsigned char *pdata) {
    memcpy(im.data, pdata, im.h * im.w * im.c);
}

Image make_image(int w, int h, int c) {
    Image out{.w=w, .h=h, .c=c, .data=0}
    out.data = (float *) xcalloc(h * w * c, sizeof(float));
    return out;
}

tk::dnn::Yolo3Detection *load_network(char *net_cfg, int n_classes, int n_batch) {
    std::string net = std::string(net_cfg);
    tk::dnn::Yolo3Detection *detNN = new tk::dnn::Yolo3Detection;
    detNN->init(net, n_classes, n_batch);
    return detNN;
}

void do_inference(tk::dnn::Yolo3Detection *net, Image im) {
    std::vector <cv::Mat> batch_dnn_input;

    cv::Mat frame(im.h, im.w, CV_8UC3, (unsigned char *) im.data);
    batch_dnn_input.push_back(frame);
    net->update(batch_dnn_input, 1);

}

Detection *get_network_boxes(tk::dnn::Yolo3Detection *net, float thresh, int batch_num, int *pnum) {
    std::vector <std::vector<tk::dnn::box>> batchDetected;
    batchDetected = net->get_batch_detected();
    int nboxes = 0;
    std::vector <std::string> classesName = net->get_classesName();
    Detection *dets = (Detection *) xcalloc(batchDetected[batch_num].size(), sizeof(Detection));

    for (int i = 0; i < batchDetected[batch_num].size(); ++i) {
        if (batchDetected[batch_num][i].prob > thresh) {
            dets[nboxes].cl = batchDetected[batch_num][i].cl;
            strcpy(dets[nboxes].name, classesName[dets[nboxes].cl].c_str());
            dets[nboxes].bbox.x = batchDetected[batch_num][i].x;
            dets[nboxes].bbox.y = batchDetected[batch_num][i].y;
            dets[nboxes].bbox.w = batchDetected[batch_num][i].w;
            dets[nboxes].bbox.h = batchDetected[batch_num][i].h;
            dets[nboxes].prob = batchDetected[batch_num][i].prob;
            nboxes += 1;
        }
    }
    if (pnum) *pnum = nboxes;
    return dets;
}
}