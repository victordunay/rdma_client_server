/* CUDA 10.2 has a bug that prevents including <cuda/atomic> from two separate
 * object files. As a workaround, we include ex2.cu directly here. */
#include "ex2.cu"
#include "common.cu"
#include <cassert>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <infiniband/verbs.h>
#include <string.h>
#include <unistd.h>


class server_rpc_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> gpu_context;

public:
    explicit server_rpc_context(uint16_t tcp_port) : rdma_server_context(tcp_port),
        gpu_context(create_queues_server(256))
    {
    }

    virtual void event_loop() override
    {
        /* so the protocol goes like this:
         * 1. we'll wait for a CQE indicating that we got an Send request from the client.
         *    this tells us we have new work to do. The wr_id we used in post_recv tells us
         *    where the request is.
         * 2. now we send an RDMA Read to the client to retrieve the request.
         *    we will get a completion indicating the read has completed.
         * 3. we process the request on the GPU.
         * 4. upon completion, we send an RDMA Write with immediate to the client with
         *    the results.
         */
        rpc_request* req;
        uchar *img_in;
        uchar *img_out;

        bool terminate = false, got_last_cqe = false;

        while (!terminate || !got_last_cqe) 
        {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) 
            {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) 
            {
		    VERBS_WC_CHECK(wc);

                switch (wc.opcode) {
                case IBV_WC_RECV:
                    /* Received a new request from the client */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];

                    /* Terminate signal */
                    if (req->request_id == -1) {
                        printf("Terminating...\n");
                        terminate = true;
                        goto send_rdma_write;
                    }

                    /* Step 2: send RDMA Read to client to read the input */
                    post_rdma_read(
                        img_in,             // local_dst
                        req->input_length,  // len
                        mr_images_in->lkey, // lkey
                        req->input_addr,    // remote_src
                        req->input_rkey,    // rkey
                        wc.wr_id);          // wr_id
                    break;

                case IBV_WC_RDMA_READ:
                    /* Completed RDMA read for a request */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];
                    img_out = &images_out[wc.wr_id * IMG_SZ];

                    // Step 3: Process on GPU
                    while(!gpu_context->enqueue(wc.wr_id, img_in, img_out)){};
		    break;
                    
                case IBV_WC_RDMA_WRITE:
                    /* Completed RDMA Write - reuse buffers for receiving the next requests */
                    post_recv(wc.wr_id % OUTSTANDING_REQUESTS);

		    if (terminate)
			got_last_cqe = true;

                    break;
                default:
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }

            // Dequeue completed GPU tasks
            int dequeued_img_id;
            if (gpu_context->dequeue(&dequeued_img_id)) {
                req = &requests[dequeued_img_id];
                img_out = &images_out[dequeued_img_id * IMG_SZ];

                //send_rdma_write:
                // Step 4: Send RDMA Write with immediate to client with the response
		        post_rdma_write(
                    req->output_addr,                       // remote_dst
                    terminate ? 0 : req->output_length,     // len
                    req->output_rkey,                       // rkey
                    terminate ? 0 : img_out,                // local_src
                    mr_images_out->lkey,                    // lkey
                    dequeued_img_id + OUTSTANDING_REQUESTS, // wr_id
                    (uint32_t*)&req->request_id);           // immediate
            }
        }
    }
};






class client_rpc_context : public rdma_client_context {
private:
    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;

    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
public:
    explicit client_rpc_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
    }

    ~client_rpc_context()
    {
        kill();
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        /* register a memory region for the input images. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        if (requests_sent - send_cqes_received == OUTSTANDING_REQUESTS)
            return false;

        struct ibv_sge sg; /* scatter/gather element */
        struct ibv_send_wr wr; /* WQE */
        struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */

        /* step 1: send request to server using Send operation */
        
        struct rpc_request *req = &requests[requests_sent % OUTSTANDING_REQUESTS];
        req->request_id = img_id;
        req->input_rkey = img_in ? mr_images_in->rkey : 0;
        req->input_addr = (uintptr_t)img_in;
        req->input_length = IMG_SZ;
        req->output_rkey = img_out ? mr_images_out->rkey : 0;
        req->output_addr = (uintptr_t)img_out;
        req->output_length = IMG_SZ;

        /* RDMA send needs a gather element (local buffer)*/
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey = mr_requests->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = img_id; /* helps identify the WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed");
            exit(1);
        }

        ++requests_sent;

        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* When WQE is completed we expect a CQE */
        /* We also expect a completion of the RDMA Write with immediate operation from the server to us */
        /* The order between the two is not guarenteed */

        struct ibv_wc wc; /* CQE */
        int ncqes = ibv_poll_cq(cq, 1, &wc);
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        if (ncqes == 0)
            return false;

	    VERBS_WC_CHECK(wc);

        switch (wc.opcode) {
        case IBV_WC_SEND:
            ++send_cqes_received;
            return false;
        case IBV_WC_RECV_RDMA_WITH_IMM:
            *img_id = wc.imm_data;
            break;
        default:
            printf("Unexpected completion type\n");
            assert(0);
        }

        /* step 2: post receive buffer for the next RPC call (next RDMA write with imm) */
        post_recv();

        return true;
    }

    void kill()
    {
        while (!enqueue(-1, // Indicate termination
                       NULL, NULL)) ;
        int img_id = 0;
        bool dequeued;
        do {
            dequeued = dequeue(&img_id);
        } while (!dequeued || img_id != -1);
    }
};


/********************************************************************************/
/*                                rdma_server_access                         */
/********************************************************************************/

struct rdma_server_remote_access
{
    uint32_t img_in_rkey
    uint64_t img_in_addr;
    uint32_t img_out_rkey
    uint64_t img_out_addr;


    uint32_t ctg_queue_rkey;
    uint64_t ctg_queue_addr;    
    uint64_t ctg_head_addr;    
    uint64_t ctg_tail_addr;    

    uint32_t gtc_queue_rkey;
    uint64_t gtc_queue_addr;
    uint64_t gtc_head_addr;
    uint64_t gtc_tail_addr;

    //uint32_t producer_index_rkey; //ctg_head
    //uint32_t producer_index_addr; //ctg_head

    int number_of_slots;

}

struct rdma_server_remote_index
{
    int ctg_head;
    int ctg_tail;
    int gtc_head;
    int gtc_tail;
}

int job_size = sizeof(Job);
int atomic_int_size = sizeof(cuda::atomic<int>);

/********************************************************************************/
/*                                server side                         */
/********************************************************************************/

class server_queues_context : public rdma_server_context 
{
private:
    std::unique_ptr<queue_server> server;

    /* TODO: add memory region(s) for CPU-GPU queues */
    
    //shared_queue *gpu_to_cpu_q;
    //shared_queue *cpu_to_gpu_q;
    struct ibv_mr *mr_gpu_to_cpu_q;
    struct ibv_mr *mr_cpu_to_gpu_q;
    int ctg_head;
    int gtc_tail;
    //struct ibv_pd *pd_cpu_to_gpu_q;
    // *pd_gpu_to_cpu_q;

   

public:
    explicit server_queues_context(uint16_t tcp_port) : rdma_server_context(tcp_port), server(create_queues_server(256))
    {
        /* TODO Initialize additional server MRs as needed. */
        struct rdma_server_remote_access rdma_server_info;
        memset(&rdma_server_info, 0, sizeof(rdma_server_info));

        gpu_to_cpu_q = server->gpu_to_cpu_q;
        cpu_to_gpu_q = server->cpu_to_gpu_q;




        mr_gpu_to_cpu_q = ibv_reg_mr(pd, server->gpu_to_cpu_q, sizeof(shared_queue), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!mr_gpu_to_cpu_q) 
        {
            fprintf(stderr, "Error, ibv_reg_mr() failed\n");
            exit(1);
        }

        mr_cpu_to_gpu_q = ibv_reg_mr(pd, server->cpu_to_gpu_q, sizeof(shared_queue), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!mr_gpu_to_cpu_q) 
        {
            fprintf(stderr, "Error, ibv_reg_mr() failed\n");
            exit(1);
        }

        rdma_server_info.img_in_rkey = mr_images_in->rkey;
        rdma_server_info.img_in_addr = (uintptr_t)images_in;
        rdma_server_info.img_out_rkey = mr_images_out->rkey;
        rdma_server_info.img_out_addr = (uintptr_t)images_out;

        rdma_server_info.ctg_queue_rkey = mr_cpu_to_gpu_q->rkey;
        rdma_server_info.ctg_queue_addr = (uintptr_t)cpu_to_gpu_q->jobs;
        rdma_server_info.ctg_head_addr = (uintptr_t)&cpu_to_gpu_q->_head;
        rdma_server_info.ctg_tail_addr = (uintptr_t)&cpu_to_gpu_q->_tail;


        rdma_server_info.gtc_queue_rkey = mr_gpu_to_cpu_q->rkey;
        rdma_server_info.gtc_queue_addr = (uintptr_t)gpu_to_cpu_q->jobs;
        rdma_server_info.ctg_head_addr = (uintptr_t)&cpu_to_gpu_q->_head;
        rdma_server_info.ctg_tail_addr = (uintptr_t)&cpu_to_gpu_q->_tail;

        rdma_server_info.num_of_slots = server->num_of_slots;
         //rdma_server_info.length = sizeof(shared_queue);


        /* TODO Exchange rkeys, addresses, and necessary information (e.g.
         * number of queues) with the client */
        send_over_socket(&rdma_server_info, sizeof(rdma_server_info));

        struct rdma_server_remote_index queue_indexes;

        queue_indexes.ctg_head = cpu_to_gpu_q->_head;
        queue_indexes.ctg_tail = cpu_to_gpu_q->_tail;
        queue_indexes.gtc_head = gpu_to_cpu_q->_head;
        queue_indexes.gtc_tail = gpu_to_cpu_q->_tail;

        send_over_socket(&queue_indexes, sizeof(queue_indexes));

    }

    ~server_queues_context()
    {
        /* TODO destroy the additional server MRs here */
        ibv_dereg_mr(mr_cpu_to_gpu_q);
        ibv_dereg_mr(mr_gpu_to_cpu_q);

    }

    void event_loop() override
    {
        /* TODO simplified version of server_rpc_context::event_loop. As the
         * client use one sided operations, we only need one kind of message to
         * terminate the server at the end. */
        rpc_request* req;
        bool terminate = false;

        while (!terminate) 
        {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);

            if (ncqes < 0) 
            {
                perror("ibv_poll_cq() failed");
                exit(1);
            }

            if (ncqes > 0) 
            {
		        VERBS_WC_CHECK(wc);

                switch (wc.opcode) 
                {
                    case IBV_WC_RECV:
                        /* Received a new request from the client */
                        req = &requests[wc.wr_id];

                        /* Terminate signal */
                        if (req->request_id == -1)
                        {
                            printf("Terminating...\n");
                            terminate = true;
                        }
                    break;
                }
            }
        }
    };
};


/********************************************************************************/
/*                                client side                      */
/********************************************************************************/

class client_queues_context : public rdma_client_context {
private:
    /* TODO add necessary context to track the client side of the GPU's
     * producer/consumer queues */
    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;
    
    //Job array for sending requests to the server + Job
    std::vector<Job> sending_jobs;
    Job recieved_job = {0,0,0};
    struct ibv_mr *mr_sending_jobs = nullptr;
    struct ibv_mr *mr_recieved_job = nullptr;


    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
   
    /* TODO define other memory regions used by the client here */
    struct rdma_server_remote_access remote_info;
    
    struct rdma_server_remote_index indexes;
    struct ibv_me *mr_indexes;
 

public:
    client_queues_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
        /* TODO communicate with server to discover number of queues, necessary
         * rkeys / address, or other additional information needed to operate
         * the GPU queues remotely. */
        recv_over_socket(&remote_info, sizeof(remote_info));
        initialize_job_pointers();
        recv_over_socket(&indexes, sizeof(indexes));
        initialize_index_pointers();
    }

    ~client_queues_context()
    {
	/* TODO terminate the server and release memory regions and other resources */
        kill();
        //need to release memory regions and other resources here    
    }

    void initialize_job_pointers()
    {
        //initialize the vector of jobs with the size of the queue.
        sending_jobs = std::vector<Job>(remote_info.num_of_slots,{0,0,0});

        //Adding memory regions
        mr_sending_jobs = ibv_reg_mr(pd, sending_jobs.begin(), sizeof(Job) * remote_info.num_of_slots , IBV_ACCESS_LOCAL_READ);
        if (!mr_sending_jobs) {
            perror("ibv_reg_mr() failed for job queue");
            exit(1);
        }
        mr_recieved_job = ibv_reg_mr(pd, &recieved_job, sizeof(Job), IBV_ACCESS_LOCAL_WRITE);
        if (!mr_recieved_job) {
            perror("ibv_reg_mr() failed for recieved_job");
            exit(1);
        }
    }

    void initialize_index_pointers()
    {
        mr_indexes = ibv_reg_mr(pd, &indexes, sizeof(indexes) , IBV_ACCESS_LOCAL_READ|IBV_ACCESS_LOCAL_WRITE);
        if (!mr_indexes) {
            perror("ibv_reg_mr() failed for indexes");
            exit(1);
        }
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        /* register a memory region for the input images. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_LOCAL_READ);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_LOCAL_READ);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        /* TODO use RDMA Write and RDMA Read operations to enqueue the task on
         * a CPU-GPU producer consumer queue running on the server. */
        
        //reading _tail from remote
        post_rdma_read(
            indexes.ctg_tail,   // local_dst
            atomic_int_size,    // len
            mr_indexes->lkey,   // lkey
            remote_info->ctg_tail_addr,    // remote_src
            remote_info->ctg_queue_rkey,    // rkey
            1);          // wr_id

        Job enqueue_job = {img_id,img_in,img_out};
        return false;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* TODO use RDMA Write and RDMA Read operations to detect the completion and dequeue a processed image
         * through a CPU-GPU producer consumer queue running on the server. */
        return false;
    }

    void kill()
    {
        while (!enqueue(-1, // Indicate termination
                       NULL, NULL)) ;
        int img_id = 0;
        bool dequeued;
        do {
            dequeued = dequeue(&img_id);
        } while (!dequeued || img_id != -1);
    }
};

std::unique_ptr<rdma_server_context> create_server(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<server_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<server_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}

std::unique_ptr<rdma_client_context> create_client(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<client_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<client_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}
