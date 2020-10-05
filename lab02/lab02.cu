#include <iostream>
#include <string>

#include "dummy_helper.hpp"
#include "image.hpp"


__device__
float to_grayscale(uchar4 pixel)
{
    float y = 0.299 * pixel.x + 0.587 * pixel.y + 0.114 * pixel.z;

    return y;
}


texture<uchar4, 2, cudaReadModeElementType> tex;

__global__
void kernel(uchar4 *out, uint32_t width, uint32_t height)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	size_t offset_x = blockDim.x * gridDim.x;
	size_t offset_y = blockDim.y * gridDim.y;

    for (size_t y = idy; y < height; y += offset_y)
        for (size_t x = idx; x < width; x += offset_x)
        {
            float  y1 = to_grayscale(tex2D(tex, x, y)),
                   y2 = to_grayscale(tex2D(tex, x+1, y)),
                   y3 = to_grayscale(tex2D(tex, x, y+1)),
                   y4 = to_grayscale(tex2D(tex, x+1, y+1));
            
            unsigned char pixel_y = sqrtf(
                (y1 - y4) * (y1 - y4) + (y2 - y3) * (y2 - y3)
            );

            out[y * width + x] = { pixel_y, pixel_y, pixel_y, 0 };
        }
}


int main()
{
    std::string input_name  = "./in.data",
                output_name = "./out.data";

    Image<uchar4> input_image(input_name);
    
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
    
    cudaArray *cuda_array;

    checkCudaErrors(cudaMallocArray(
        &cuda_array,
        &channel_desc,
        input_image.width,
        input_image.height
    ));

    checkCudaErrors(cudaMemcpyToArray(
        cuda_array,
        0,
        0,
        input_image.buffer.data(),
        input_image.size(),
        cudaMemcpyHostToDevice
    ));

    tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = channel_desc;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false;

    checkCudaErrors(cudaBindTextureToArray(
        tex,
        cuda_array,
        channel_desc
    ));
    
    CudaMemory<uchar4> output_image_buffer_d(input_image.count());

    kernel<<<dim3(4, 4), dim3(16, 16)>>>(
        output_image_buffer_d.get(), 
        input_image.width, 
        input_image.height
    );

    Image<uchar4> output_image = input_image;

    output_image_buffer_d.memcpy(
        output_image.buffer.data(),
        cudaMemcpyDeviceToHost
    );

    output_image.save(output_name);

    checkCudaErrors(cudaUnbindTexture(tex));

    checkCudaErrors(cudaFreeArray(cuda_array));

    return 0;
}
