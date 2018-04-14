#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include<time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include "device_launch_parameters.h"
#include "GpuTimer.h"

using namespace cv;
using namespace std;

// cpu implementation
void rgb2grayCPU(unsigned char* color, unsigned char* gray, int numRows, int numCols, int numChannels) {
    int grayOffset, colorOffset;

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            // linearize pixel coordinate tuple (i, j)
            grayOffset = i * numCols + j;
            colorOffset = grayOffset * numChannels;

            // convert to gray
            gray[grayOffset] = (0.21 * color[colorOffset + 2]) +
                               (0.71 * color[colorOffset + 1]) +
                               (0.07 * color[colorOffset]);
       }
   }
}

// gpu implementation
__global__ void rgb2grayGPU(unsigned char* Pout, unsigned char* Pin, int width, int height, int numChannels) {
    // compute global thread coordinates
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;

    // linearize coordinates for data access
    int grayOffset = row * width + col;
    int colorOffset = grayOffset * numChannels;

    if ((col < width) && (row < height)) {
        Pout[grayOffset] = (0.21 * Pin[colorOffset + 2]) +
                           (0.71 * Pin[colorOffset + 1]) +
                           (0.07 * Pin[colorOffset]);
    }
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("[!] Filename expected.\n");
        return 0;
    }

    // read image
    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
        printf("Cannot read image file %s", argv[1]);
        exit(1);
    }

    // define img params and timers
    int imageChannels = 3;
    int imageWidth = image.cols;
    int imageHeight = image.rows;
    size_t size_rgb = sizeof(unsigned char)*imageWidth*imageHeight*imageChannels;
    size_t size_gray = sizeof(unsigned char)*imageWidth*imageHeight;
    GpuTimer timer, timer1, timer2, timer3;

    // allocate mem for host image vectors
    unsigned char* h_grayImage = (unsigned char*)malloc(size_rgb);
    unsigned char* h_grayImage_CPU = (unsigned char*)malloc(size_rgb);

    // grab pointer to host rgb image
    unsigned char* h_rgbImage = image.data;

    // allocate mem for device rgb and gray
    unsigned char* d_rgbImage;
    unsigned char* d_grayImage;
    timer.Start();
    cudaMalloc((void**)&d_rgbImage, size_rgb);
    cudaMalloc((void**)&d_grayImage, size_gray);
    timer.Stop();
    float d_t0 = timer.Elapsed();
    printf("Time to allocate memory on the device is: %f msecs.\n", d_t0);

    // copy the rgb image from the host to the device and record the needed time
    timer1.Start();
    cudaMemcpy(d_rgbImage, h_rgbImage, size_rgb, cudaMemcpyHostToDevice);
    timer1.Stop();
    float d_t1 = timer1.Elapsed();
    printf("Time to copy the RGB image from the host to the device is: %f msecs.\n", d_t1);

    // execution configuration parameters + kernel launch
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(imageWidth/16.0), ceil(imageHeight/16.0), 1);
    timer2.Start();
    rgb2grayGPU<<<dimGrid, dimBlock>>>(d_grayImage, d_rgbImage, imageWidth, imageHeight, imageChannels);
    timer2.Stop();
    float d_t2 = timer2.Elapsed();
    printf("Implemented CUDA code ran in: %f msecs.\n", d_t2);

    // copy gray image from device to host
    timer3.Start();
    cudaMemcpy(h_grayImage, d_grayImage, size_gray, cudaMemcpyDeviceToHost);
    timer3.Stop();
    float d_t3 = timer.Elapsed();
    printf("Time to copy the Gray image from the device to the host is: %f msecs.\n", d_t3);

    // do the processing on the CPU
    clock_t begin = clock();
    rgb2grayCPU(h_rgbImage, h_grayImage_CPU, imageHeight, imageWidth, imageChannels);
    clock_t end = clock();

    // calculate total time for CPU and GPU
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC*1000;
    printf("Total CPU code ran in: %f msecs.\n", time_spent);
    float total_device_time = d_t0 + d_t1 + d_t2 + d_t3;
    printf("Total GPU code ran in: %f msecs.\n", total_device_time);
    float speedup = (float)time_spent / total_device_time;
    printf("GPU Speedup: %f\n", speedup);

    // display images
    Mat Image1(imageHeight, imageWidth, CV_8UC1, h_grayImage);
    Mat Image2(imageHeight, imageWidth, CV_8UC1, h_grayImage_CPU);
    namedWindow("CPUImage", WINDOW_NORMAL);
    namedWindow("GPUImage", WINDOW_NORMAL);
    imshow("GPUImage",Image1);
    imshow("CPUImage",Image2);
    waitKey(0);

    // free host and device memory
    image.release();
    Image1.release();
    Image2.release();
    free(h_grayImage);
    free(h_grayImage_CPU);
    cudaFree(d_rgbImage); cudaFree(d_grayImage);

    return 0;
}
