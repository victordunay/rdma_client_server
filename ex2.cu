/* This file should be almost identical to ex2.cu from homework 2. */
/* once the TODOs in this file are complete, the RPC version of the server/client should work correctly. */

#include "ex3.h"
#include "ex2.h"
#include <cuda/atomic>

#define N_STREAMS (64)
#define N_BINS (256)
#define N_TB_SERIAL (1)
#define N_THREADS_Y (16)
#define N_THREADS_X (64)
#define N_THREADS_Z (1)
#define NO_EMPTY_STREAMS (-1)
#define INIT_ID (-1)
#define KILLING_JOB (-1)
#define SHARED_MEM_PER_BLOCK (2048)
#define REGISTERS_PER_THREAD (32)
#define DEVICE (0)


typedef cuda::atomic<int,cuda::thread_scope_device> atomic_lock_t;

 /**
  * @brief Create a histogram of the tile pixels. Assumes that the kernel runs with more than 256 threads
  * 
  * @param image_start The start index of the image the block processing
  * @param t_row  The tile's row
  * @param t_col  The tile's column
  * @param histogram - the histogram of the tile
  * @param image - The images array to process.
  */
 __device__ void create_histogram(int image_start, int t_row, int t_col ,int *histogram, uchar *image)
 {
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // initialize histogram
    if(tid < N_BINS)
    {
        histogram[tid] = 0;
    }
    __syncthreads();
    //calculates the pixel index that assigned to the thread 
    int row_base_offset = (t_row * TILE_WIDTH + threadIdx.y) * IMG_WIDTH ;
    int row_interval = blockDim.y * IMG_WIDTH;
    int col_offset = t_col * TILE_WIDTH + threadIdx.x; 

    uchar pixel_value = 0;

    //The block has 16 rows, Therefore, it runs 4 times so every warp run on 4 different rows
    for(int i = 0; i < TILE_WIDTH/blockDim.y; i++ ) 
    {
        pixel_value = image[image_start + row_base_offset + (i * row_interval) + col_offset];
        atomicAdd(&(histogram[pixel_value]), 1);
    } 
 }


 /**
  * @brief Calculates inclusive prefix sum of the given array. Saves the sum in the given array.
  *      Assumes n_threads > arr_size
  * 
  * @param arr The given array 
  * @param arr_size The size of the array
  */
__device__ void prefix_sum(int arr[], int arr_size) 
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int increment;
    for (int stride = 1; stride < arr_size; stride *= 2) 
    {
        if (tid >= stride && tid < arr_size) 
                increment = arr[tid - stride];
        __syncthreads(); 
        if (tid >= stride && tid < arr_size) 
                arr[tid] += increment;
        __syncthreads();
    }
    return;
}


/**
 * @brief Calculates a map from the cdf and saves it in the given index in the 'maps' array.
 * 
 * @param map_start The start index in the 'maps' array of the current image's map
 * @param t_row The tile's row
 * @param t_col The tile's column
 * @param cdf The cdf of the tile.
 * @param maps Array of the maps of all images
 * @return __device__ 
 */
__device__ void calculate_maps(int map_start, int t_row, int t_col, int *cdf, uchar *maps)
{
    uchar div_result = (uchar) 0;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < N_BINS)
    {
        div_result = (uchar)(cdf[tid] * 255.0/(TILE_WIDTH*TILE_WIDTH));
        maps[map_start + (t_col + t_row * TILE_COUNT)*N_BINS + tid] = div_result;
    }   
    __syncthreads();     
}
/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
 void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);


/**
 * @brief process an image which assigned to the block index. It takes an image given in all_in, and return the processed image in all_out respectively.
 * 
 * @param in Array of input images, in global memory ([N_IMAGES][IMG_HEIGHT][IMG_WIDTH])
 * @param all_out Array of output images, in global memory ([N_IMAGES][IMG_HEIGHT][IMG_WIDTH])
 * @param maps 4D array ([N_IMAGES][TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @return __global__ 
 */
__device__ void process_image(uchar *in, uchar *out, uchar* maps) 
{
   __shared__ int cdf[N_BINS];
    int image_start = IMG_WIDTH * IMG_HEIGHT * blockIdx.x;
    int map_start = TILE_COUNT * TILE_COUNT * N_BINS * blockIdx.x;
    for(int t_row = 0; t_row< TILE_COUNT; ++t_row)
    {
        for(int t_col = 0; t_col< TILE_COUNT; ++t_col)
        {
            create_histogram(image_start,t_row, t_col, cdf, in);
            __syncthreads();
            prefix_sum(cdf, N_BINS);
            calculate_maps(map_start, t_row, t_col,cdf, maps); 
            __syncthreads();
        }
    }
    interpolate_device(&maps[map_start],&in[image_start], &out[image_start]);
    return; 

}
// TODO complete according to HW2:
//          implement a lock,
//          implement a MPMC queue,
//          implement the persistent kernel,
//          implement a function for calculating the threadblocks count
// (This file should be almost identical to ex2.cu from homework 2.)


/*********************************************************************************************************/
/*                                          locks                                                        */
/*********************************************************************************************************/ 

__device__ atomic_lock_t* _readerlock;
__device__ atomic_lock_t* _writerlock;

__device__  void Lock(atomic_lock_t* _lock) 
{
    while(_lock->load(cuda::memory_order_relaxed) == 1);
    while(_lock->exchange(1, cuda::memory_order_acq_rel));
}

__device__  void Unlock(atomic_lock_t * _lock) 
{
    _lock->store(0, cuda::memory_order_release);
}

__global__  void Initlocks() 
{
    _readerlock = new atomic_lock_t(0);
    _writerlock = new atomic_lock_t(0);
}

__global__ void FreeLocks()
{
    delete _readerlock;
    delete _writerlock;
}



//A single item in the queue
struct Job
{
    int img_id;
    uchar* img_in;
    uchar* img_out;
};


/*********************************************************************************************************/
/*                                      queue (ring buffer) class                                        */
/*********************************************************************************************************/

class shared_queue 
{
private:

    //queue data
    size_t queue_size;
    
    
public:
    //queue sync variables
    cuda::atomic<int> _head;
    cuda::atomic<int> _tail;
    //queue of jobs;
    Job* jobs;
       
    /**
    * @brief enqueue an image by a job and adding it to the queue (if not full). sending -1 by the cpu is for terminate the kernel
    * 
    * @param enqueue_job a Job struct. send to be enqueued
    * @return true if enqueue succeeded and false otherwise (full)
    */
    __device__  __host__ bool enqueueJob(Job enqueue_job)
    {
        int tail =  _tail.load(cuda::memory_order_relaxed);
        if(tail - (int)_head.load(cuda::memory_order_acquire) != (int)queue_size)
        {
            jobs[tail % queue_size] = enqueue_job;
            _tail.store(tail + 1, cuda::memory_order_release);
            return true;
        }
        return false;
    }

    /**
    * @brief dequeue a job (if not empty) and saves it to the Job pointer given
    * 
    * @param dequeue_job a pointer to Job struct. send to be dequeued
    * @return true if dequeue succeeded and false otherwise (empty)
    */
    __device__  __host__ bool dequeueJob(Job* dequeue_job)
    {
        int head = _head.load(cuda::memory_order_relaxed);
        if(_tail.load(cuda::memory_order_acquire) != head)
        {
            *dequeue_job = jobs[head % queue_size];
            _head.store(head + 1, cuda::memory_order_release);
            return true;
        }  
        return false;
    }

    //constructor creat an array of Jobs in size of queue_size jobs
    shared_queue(int queue_size): queue_size(queue_size), jobs(nullptr), _head(0),_tail(0)
    {   
        size_t size_int_in_bytes = queue_size * sizeof(Job);
        CUDA_CHECK(cudaMallocHost(&jobs, size_int_in_bytes));
    }

    //destructor frees the array first
    ~shared_queue() 
    {
        CUDA_CHECK(cudaFreeHost(jobs));
    }
};


/*********************************************************************************************************/
/*                                      the consumer_producer kernel                                     */
/*********************************************************************************************************/

__global__
void consumer_proccessor(shared_queue *gpu_to_cpu_q,shared_queue *cpu_to_gpu_q, uchar* maps)
{
    __shared__ Job job;
    int tid = threadIdx.y + threadIdx.x + threadIdx.z;

    //dequeue the first job
    if(tid == 0)
    {
        Lock(_readerlock);
        while(!cpu_to_gpu_q->dequeueJob(&job));
        Unlock(_readerlock);
    }
    __syncthreads();
    
    //kernel runs until it get img_id = KILLING_JOB
    while(job.img_id != KILLING_JOB)
    {
        process_image(job.img_in, job.img_out, maps);
        __syncthreads();
        if(tid == 0)//enqueue the finished job back to cpu
        {
            Lock(_writerlock);
            while(!gpu_to_cpu_q->enqueueJob(job));
            Unlock(_writerlock);
        }
        __syncthreads();
        if(tid == 0 )//dequeue new job from the cpu
        {
            Lock(_readerlock);
            while(!cpu_to_gpu_q->dequeueJob(&job));
            Unlock(_readerlock);
        }
        __syncthreads();
    }
}
    
/*********************************************************************************************************/
/*                                      queue_server class                                               */
/*********************************************************************************************************/

class queue_server : public image_processing_server
{

private:
    // TODO define queue server context (memory buffers, etc...)
   
    int threadblocks;
    char* cpu_to_gpu_buffer;
    char* gpu_to_cpu_buffer;
    uchar *maps;

    /**
     * @brief calculates the number of thread blocks the device can handle in parrallel
     * 
     * @param threads the number of threads in each threadblock
     * @return int the number of threadblocks
     */
    int calcNumOfTB(int threads)
    {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, DEVICE));

        int num_of_multi = prop.multiProcessorCount;
        int threads_per_multi = prop.maxThreadsPerMultiProcessor ;
        int regs_per_multi = prop.regsPerMultiprocessor;
        int shared_per_multi = prop.sharedMemPerMultiprocessor;
        
        int threads_bound = (int) (floor(threads_per_multi/threads)*num_of_multi);
        int regs_bound = (int) (floor(regs_per_multi/(REGISTERS_PER_THREAD*threads))*num_of_multi);
        int shared_mem_bound = (int) (floor(shared_per_multi/SHARED_MEM_PER_BLOCK)*num_of_multi);

        //calc min in a stupid way
        if(threads_bound > shared_mem_bound)
        {
            return (shared_mem_bound>regs_bound) ? regs_bound : shared_mem_bound;
        }
        else
        {
            return (threads_bound>regs_bound) ? regs_bound : threads_bound;
        }
        return -1;
    }


 

public:

    shared_queue *gpu_to_cpu_q;
    shared_queue *cpu_to_gpu_q;
    int num_of_slots;
    queue_server(int threads)
    {
        // TODO initialize host state
        threadblocks = calcNumOfTB(threads);
        num_of_slots =(int) (pow(2,ceil(log2(16*threadblocks))));

        //allocate 2 pined queues
        CUDA_CHECK(cudaMallocHost(&cpu_to_gpu_buffer, sizeof(shared_queue)));
        CUDA_CHECK(cudaMallocHost(&gpu_to_cpu_buffer, sizeof(shared_queue)));
        cpu_to_gpu_q = new (cpu_to_gpu_buffer) shared_queue(num_of_slots);
        gpu_to_cpu_q = new (gpu_to_cpu_buffer)  shared_queue(num_of_slots);
    
        //initiate locks
        Initlocks<<<1,1>>>();

        //initiate data for proccessing - allocating arrays of data like in bulk for temp use.
        CUDA_CHECK( cudaMalloc(&maps, threadblocks * TILE_COUNT * TILE_COUNT * N_BINS) );
        //kernel invocing
        dim3 GRID_SIZE(N_THREADS_X, threads/N_THREADS_X , N_THREADS_Z);
        consumer_proccessor<<<threadblocks, GRID_SIZE>>>(gpu_to_cpu_q,cpu_to_gpu_q,maps);
    }

    ~queue_server() override
    {
        //kills all threadblocks
        for(int i= 0; i<threadblocks; ++i)
        {
            Job killing_job = {KILLING_JOB,nullptr,nullptr};
            cpu_to_gpu_q->enqueueJob(killing_job);
        }
        //free locks
        FreeLocks<<<1,1>>>();
        CUDA_CHECK(cudaDeviceSynchronize());

        // free queues memory
        gpu_to_cpu_q->~shared_queue();
        cpu_to_gpu_q->~shared_queue();
        CUDA_CHECK(cudaFreeHost(cpu_to_gpu_buffer));
        CUDA_CHECK(cudaFreeHost(gpu_to_cpu_buffer));
        CUDA_CHECK( cudaFree(maps));
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        Job enqueue_job = {img_id,img_in,img_out};
        return cpu_to_gpu_q->enqueueJob(enqueue_job);
    }

    bool dequeue(int *img_id) override
    {
        Job dequeue_job;
        bool ans = gpu_to_cpu_q->dequeueJob(&dequeue_job);
        if(ans)
            *img_id = dequeue_job.img_id;
        return ans;
    }
};

std::unique_ptr<queue_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
};