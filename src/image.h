/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2018/06/07
 */
#ifndef __IMAGE_H_
#define __IMAGE_H_

#include <opencv2/opencv.hpp>
using namespace cv;

typedef struct
{
    int w;
    int h;
    int c;
    float *data;
}image;

image make_image(int w, int h, int c);

image make_empty_image(int w, int h, int c);

image load_image_resize(char *filename, int w, int h, int c, image *im);

image load_mat_resize(Mat mat,int w, int h, int c, image *im);

image load_image_color(char* filename,int w,int h);

void free_image(image m);

image letterbox_image(image im, int w, int h);

static float get_pixel(image m, int x, int y, int c);

static void set_pixel(image m, int x, int y, int c, float val);

static void add_pixel(image m, int x, int y, int c, float val);

#endif
