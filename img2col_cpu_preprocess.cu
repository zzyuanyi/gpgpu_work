#include<stdio.h>
#include<time.h>
#define DATA_WIDTH 16384
#define COVN_WIDTH 2
__constant__ int conv[COVN_WIDTH*COVN_WIDTH];//Convolution kernel placed in the constant region
__global__ void img2col_cau(int* in,int * out)//kernel-function
{
int bidx=blockIdx.x;
int thdx=threadIdx.x;
int bck_size=blockDim.x;
int sum=0;
while(thdx<(DATA_WIDTH-COVN_WIDTH+1))//conv_calu
{
    for(int i=0;i<COVN_WIDTH*COVN_WIDTH;i++)
{sum+=in[(bidx*(DATA_WIDTH-COVN_WIDTH+1)+thdx)*4+i]*conv[i];}
out[(bidx*(DATA_WIDTH-COVN_WIDTH+1)+thdx)]=sum;
sum=0;
thdx+=bck_size;
}
__syncthreads();

}
int main()
{
    int N=DATA_WIDTH*DATA_WIDTH;
    int M=COVN_WIDTH*COVN_WIDTH;
    int *h_data_in,*h_data_use,*h_data_out,*h_conv_in,*d_data_use,*d_data_out,*d_conv_use;
    h_data_in=(int*)malloc(sizeof(int)*N);
    h_conv_in=(int*)malloc(sizeof(int)*M);
    h_data_use=(int*)malloc(sizeof(int)*M*(DATA_WIDTH-COVN_WIDTH+1)*(DATA_WIDTH-COVN_WIDTH+1));
    h_data_out=(int*)malloc(sizeof(int)*(DATA_WIDTH-COVN_WIDTH+1)*(DATA_WIDTH-COVN_WIDTH+1));
    for(int i=0;i<DATA_WIDTH;i++)
    {
        for(int j=0;j<DATA_WIDTH;j++)
        {
            h_data_in[i*DATA_WIDTH+j]=2;
        }
    }
    for(int i=0;i<COVN_WIDTH;i++)
    {
        for(int j=0;j<COVN_WIDTH;j++)
        {
            h_conv_in[i*COVN_WIDTH+j]=3;
        }
    }
    // cudaEvent_t time_start,time_stop;
    // cudaEventCreate(&time_start);
    // cudaEventCreate(&time_stop);
    // cudaEventRecord(time_start,0);
    // cudaEventSynchronize(time_start);
    cudaMalloc((void**)&d_conv_use,sizeof(int)*M);
    cudaMalloc((void**)&d_data_use,sizeof(int)*M*(DATA_WIDTH-COVN_WIDTH+1)*(DATA_WIDTH-COVN_WIDTH+1));
    cudaMalloc((void**)&d_data_out,sizeof(int)*(DATA_WIDTH-COVN_WIDTH+1)*(DATA_WIDTH-COVN_WIDTH+1));
    for(int i=0;i<(DATA_WIDTH-COVN_WIDTH+1);i++)//data_preprocess
    {
        for(int j=0;j<(DATA_WIDTH-COVN_WIDTH+1);j++)
        {
            h_data_use[(i*(DATA_WIDTH-COVN_WIDTH+1)+j)*4]=h_data_in[(i*(DATA_WIDTH-COVN_WIDTH+1)+j)];
            h_data_use[(i*(DATA_WIDTH-COVN_WIDTH+1)+j)*4+1]=h_data_in[(i*(DATA_WIDTH-COVN_WIDTH+1)+j+1)];
            h_data_use[(i*(DATA_WIDTH-COVN_WIDTH+1)+j)*4+2]=h_data_in[((i+1)*(DATA_WIDTH-COVN_WIDTH+1)+j)];
            h_data_use[(i*(DATA_WIDTH-COVN_WIDTH+1)+j)*4+3]=h_data_in[((i+1)*(DATA_WIDTH-COVN_WIDTH+1)+j+1)];
        }
    }
    cudaMemcpy(d_data_use,h_data_use,sizeof(int)*M*(DATA_WIDTH-COVN_WIDTH+1)*(DATA_WIDTH-COVN_WIDTH+1),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(conv,h_conv_in,sizeof(int)*M);


    img2col_cau<<<(DATA_WIDTH-COVN_WIDTH+1),64>>>(d_data_use,d_data_out);
    cudaDeviceSynchronize();

    // cudaEventRecord(time_stop,0);
    // cudaEventSynchronize(time_stop);
    // float elapsedTime;
    // cudaEventElapsedTime(&elapsedTime,time_start,time_stop);
    // printf("runtime:%f(ms)\n",elapsedTime);

    cudaMemcpy(h_data_out,d_data_out,sizeof(int)*(DATA_WIDTH-COVN_WIDTH+1)*(DATA_WIDTH-COVN_WIDTH+1),cudaMemcpyDeviceToHost);
// for(int i=0;i<(DATA_WIDTH-COVN_WIDTH+1);i++)
// {
//     for(int j=0;j<(DATA_WIDTH-COVN_WIDTH+1);j++)
//     {
//         printf(" %d ",h_data_out[i*(DATA_WIDTH-COVN_WIDTH+1)+j]);
//     }
//     printf("\n");
// }

    cudaFree(d_conv_use);
    cudaFree(d_data_out);
    cudaFree(d_data_use);
    free(h_data_in);
    free(h_data_use);
    free(h_conv_in);
    free(h_data_out);

}
