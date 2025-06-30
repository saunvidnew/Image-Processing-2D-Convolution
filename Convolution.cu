#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <wb.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#define MASK_WIDTH 5
#define O_TILE_WIDTH 16
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE 
//implement the tiled 2D convolution kernel with adjustments for channels and make sure to:
//-use the constant memory for the convolution mask
//-use shared memory to reduce the number of global accesses and handle the boundary conditions when loading input list elements into the shared memory
//-clamp your output values

__global__ void imageConvolution(float *deviceInputImageData, float *deviceOutputImageData,const float* __restrict__ deviceMaskData, int imageChannels, 
    int imageWidth, int imageHeight) {

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row_o = by * O_TILE_WIDTH + ty;
    int col_o = bx * O_TILE_WIDTH + tx;

    __shared__ float Ns[O_TILE_WIDTH+MASK_WIDTH-1][O_TILE_WIDTH + MASK_WIDTH - 1][3];

    for (int ch = 0; ch < 3; ch++) {

        int row_i = row_o - MASK_WIDTH / 2;
        int col_i = col_o - MASK_WIDTH / 2;

        if (row_i > -1 && row_i < imageHeight && col_i<imageWidth && col_i>-1) {

            Ns[ty][tx][ch] = deviceInputImageData[(row_i*imageWidth+col_i)*imageChannels+ch];
        }
        else {

            Ns[ty][tx][ch] = 0;
        }
    }
    __syncthreads();

    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {

        for (int ch = 0; ch < 3; ch++) {

            float pvalue = 0;
            for (int i = 0; i < MASK_WIDTH; i++) {

                for (int j = 0; j < MASK_WIDTH; j++) {

                    pvalue += deviceMaskData[i * MASK_WIDTH + j] * Ns[ty+i][tx+j][ch];

                }
            }

            if (row_o < imageHeight && col_o < imageWidth) {
                deviceOutputImageData[(row_o * imageWidth + col_o) * imageChannels + ch] = clamp(pvalue,0.0,1.0);
            }
        }


    }


}

int main(int argc, char* argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char* inputImageFile;
    char* inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float* hostInputImageData;
    float* hostOutputImageData;
    float* hostMaskData;
    float* deviceInputImageData;
    float* deviceOutputImageData;
    float* deviceMaskData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float*)wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == MASK_WIDTH);    /* mask height is fixed to 5 */
    assert(maskColumns == MASK_WIDTH); /* mask width is fixed to 5 */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    //@@ INSERT CODE HERE
    //allocate device memory
    cudaMalloc((void**)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void**)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void**)&deviceMaskData, maskRows*maskColumns*sizeof(float));

    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    //@@ INSERT CODE HERE
    //copy host memory to device
    cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    //initialize thread block and kernel grid dimensions
    //invoke CUDA kernel
    dim3 dimGrid((imageWidth - 1) / O_TILE_WIDTH + 1, (imageHeight - 1) / O_TILE_WIDTH + 1, 1);
    
    int BLOCK_WIDTH = O_TILE_WIDTH+MASK_WIDTH-1;
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    imageConvolution<< <dimGrid, dimBlock >> > (deviceInputImageData, deviceOutputImageData, deviceMaskData, imageChannels, imageWidth, imageHeight);
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    //@@ INSERT CODE HERE
    //copy results from device to host	
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(arg, outputImage);

    //@@ INSERT CODE HERE
    //deallocate device memory	
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
