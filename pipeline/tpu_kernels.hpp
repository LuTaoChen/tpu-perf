#ifndef TPUKERNEL_H
#define TPUKERNEL_H

#include <map>
#include <memory>
#include "bmruntime_interface.h"
#include "SGDevicePool.h"
#include "SGLog.h"
#include "bmlib_runtime.h"
#include <iostream>
#include <fstream>

using namespace std;

typedef unsigned long long u64;
#define MAX_YOLO_INPUT_NUM 3
#define MAX_YOLO_ANCHOR_NUM 3
typedef struct {
    u64 bottom_addr[MAX_YOLO_INPUT_NUM];
    u64 top_addr;
    u64 detected_num_addr;
    int input_num;
    int batch_num;
    int hw_shape[MAX_YOLO_INPUT_NUM][2];
    int num_classes;
    int num_boxes;
    int keep_top_k;
    float nms_threshold;
    float confidence_threshold;
    float bias[MAX_YOLO_INPUT_NUM * MAX_YOLO_ANCHOR_NUM * 2];
    float anchor_scale[MAX_YOLO_INPUT_NUM];
    int clip_box;
    int agnostic_nms;
}__attribute__((packed)) tpu_kernel_api_yolov5NMS_t;


#define MAX_YOLO_INPUT_NUM 3
#define MAX_YOLO_ANCHOR_NUM 3
typedef struct {
    u64 bottom_addr;
    u64 top_addr;
    u64 detected_num_addr;
    int input_shape[3];
    int keep_top_k;
    float nms_threshold;
    float confidence_threshold;
    int agnostic_nms;
    int max_hw;
}__attribute__((packed)) tpu_kernel_api_yolov5NMS_v2_t;


namespace bm {

    /**
     * @todo: add config interface(thresholds, anchors ...)
    */
    int yolov5_decode(int outNum, const TensorVec &outTensors, OutputType &postOut, ContextPtr ctx)
    {
        tpu_kernel_function_t func_id;
        func_id = ctx->getKernelFuncId();

        // bm_profile_t start, end;
        // memset(&start, 0, sizeof(start));
        // memset(&end, 0, sizeof(end));
        // bm_get_profile(ctx->handle, &start);

        int out_len_max = 200 * 7;
        int batch_num = ctx->batchSize;
        float bias[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62,
                          45, 59, 119, 116, 90, 156, 198, 373, 326};
        float downsample[3] = {8, 16, 32};
        int num_anchors = 3;
        float nms_threshold = 0.5;
        float confidence_threshold = 0.1;
        int keep_top_k = 200;
        bm_device_mem_t in_dev_mem[outNum];
        bm_device_mem_t out_dev_mem;
        bm_device_mem_t detect_num_mem;
        postOut.num = 2;
        postOut.tensors = new tensor_data_t[2];

        tensor_data_t &output_tensor = postOut.tensors[0];
        tensor_data_t &detect_num = postOut.tensors[1];
        auto output_data = new float[batch_num * keep_top_k * 7];
        output_tensor.dtype = BM_FLOAT32;
        output_tensor.shape[0] = 1;
        output_tensor.shape[1] = 1;
        output_tensor.shape[2] = 1;
        output_tensor.shape[3] = 7;
        output_tensor.dims = 4;
        output_tensor.data = reinterpret_cast<uint8_t *>(output_data);

        auto dt_num_data = new int32_t[batch_num];
        detect_num.dtype = BM_UINT32;
        detect_num.shape[0] = batch_num;
        detect_num.dims = 1;
        detect_num.data = reinterpret_cast<uint8_t *>(dt_num_data);

        bm_status_t ret = BM_SUCCESS;

        ret = bm_malloc_device_byte(ctx->handle, &out_dev_mem, out_len_max * sizeof(float));
        assert(BM_SUCCESS == ret);
        ret = bm_malloc_device_byte(ctx->handle, &detect_num_mem, batch_num * sizeof(int32_t));
        assert(BM_SUCCESS == ret);

        //  ============================ config ============================
        tpu_kernel_api_yolov5NMS_t api;
        for (int i = 0; i < outNum; i++)
        {
            in_dev_mem[i] = *outTensors[i]->get_device_mem();
            api.bottom_addr[i] = bm_mem_get_device_addr(in_dev_mem[i]);
        }
        api.top_addr = bm_mem_get_device_addr(out_dev_mem);
        api.detected_num_addr = bm_mem_get_device_addr(detect_num_mem);
        api.input_num = outNum;
        api.batch_num = batch_num;
        for (int i = 0; i < outNum; ++i)
        {
            api.hw_shape[i][0] = outTensors[i]->shape(2);
            api.hw_shape[i][1] = outTensors[i]->shape(3);
        }

        api.num_classes = outTensors[0]->shape(1) / num_anchors - 5;
        api.num_boxes = num_anchors;
        api.keep_top_k = keep_top_k;
        api.nms_threshold = nms_threshold;
        api.confidence_threshold = confidence_threshold;
        memcpy((void *)api.bias, bias, 18 * sizeof(float));
        for (int i = 0; i < outNum; ++i)
        {
            api.anchor_scale[i] = downsample[i];
        }
        api.clip_box = 0;
        api.agnostic_nms = 1;

        // std::chrono::steady_clock::time_point starts;
        // starts = std::chrono::steady_clock::now();

        tpu_kernel_launch(ctx->handle, func_id, &api, sizeof(api));
        bm_thread_sync(ctx->handle);

        // std::chrono::steady_clock::time_point ends;
        // ends = std::chrono::steady_clock::now();

        bm_memcpy_d2s_partial_offset(ctx->handle,
                                     (void *)dt_num_data,
                                     detect_num_mem,
                                     batch_num * sizeof(int32_t),
                                     0);
        output_tensor.shape[2] = *dt_num_data;
        if (*dt_num_data != 0) {
            bm_memcpy_d2s_partial_offset(ctx->handle,
                                         (void *)output_data,
                                         out_dev_mem,
                                         output_tensor.shape[2] * output_tensor.shape[3] * sizeof(float),
                                         0);
        }

        // bm_get_profile(ctx->handle, &end);

        // size_t npu_time = end.tpu_process_time - start.tpu_process_time;
        // size_t cdma_time = end.cdma_in_time - start.cdma_in_time + end.cdma_out_time - start.cdma_out_time;
        // std::cout << "npu time = " << npu_time << "(us)" << endl;
        // std::cout << "cdma time = " << cdma_time << "(us)" << endl;
        // std::cout << "time = " << std::chrono::duration_cast<std::chrono::microseconds>(ends - starts).count() << endl;
        return 0;
    }

    int yolov5_without_decode(int outNum, const TensorVec &outTensors, OutputType &postOut, ContextPtr ctx)
    {
        tpu_kernel_function_t func_id;
        func_id = ctx->getKernelFuncId();

        // bm_profile_t start, end;
        // memset(&start, 0, sizeof(start));
        // memset(&end, 0, sizeof(end));
        // bm_get_profile(ctx->handle, &start);

        int out_len_max = 200 * 7;
        int batch_num = ctx->batchSize;
        int input_shape[3] = {1, 25200, 85};
        int num_anchors = 3;
        float nms_threshold = 0.5;
        float confidence_threshold = 0.001;
        int keep_top_k = 200;
        bm_device_mem_t in_dev_mem;
        bm_device_mem_t out_dev_mem;
        bm_device_mem_t detect_num_mem;
        postOut.num = 2;
        postOut.tensors = new tensor_data_t[2];

        tensor_data_t &output_tensor = postOut.tensors[0];
        tensor_data_t &detect_num = postOut.tensors[1];
        auto output_data = new float[batch_num * keep_top_k * 7];
        output_tensor.dtype = BM_FLOAT32;
        output_tensor.shape[0] = 1;
        output_tensor.shape[1] = 1;
        output_tensor.shape[2] = 1;
        output_tensor.shape[3] = 7;
        output_tensor.dims = 4;
        output_tensor.data = reinterpret_cast<uint8_t *>(output_data);

        auto dt_num_data = new int32_t[batch_num];
        detect_num.dtype = BM_UINT32;
        detect_num.shape[0] = batch_num;
        detect_num.dims = 1;
        detect_num.data = reinterpret_cast<uint8_t *>(dt_num_data);

        bm_status_t ret = BM_SUCCESS;

        ret = bm_malloc_device_byte(ctx->handle, &out_dev_mem, out_len_max * sizeof(float));
        assert(BM_SUCCESS == ret);
        ret = bm_malloc_device_byte(ctx->handle, &detect_num_mem, batch_num * sizeof(int32_t));
        assert(BM_SUCCESS == ret);

        //  ============================ config ============================
        tpu_kernel_api_yolov5NMS_v2_t api;
        in_dev_mem = *outTensors[0]->get_device_mem();
        api.bottom_addr = bm_mem_get_device_addr(in_dev_mem);
        api.top_addr = bm_mem_get_device_addr(out_dev_mem);
        api.detected_num_addr = bm_mem_get_device_addr(detect_num_mem);
        api.keep_top_k = keep_top_k;
        api.nms_threshold = nms_threshold;
        api.confidence_threshold = confidence_threshold;
        memcpy((void *)api.input_shape, input_shape, 3 * sizeof(int));
        api.agnostic_nms = 0;
        api.max_hw = 640;


        // auto model_output = new float[batch_num * 25200 * 85];
        // bm_memcpy_d2s_partial_offset(ctx->handle,
        //                              (void *)model_output,
        //                              in_dev_mem,
        //                              batch_num * 25200 * 85 * sizeof(float),
        //                              0);
        // ofstream outfile("/mnt/onager/source/yolov5/binfile.bin", ios::out | ios::binary);  
        // if (!outfile) {
        //     cerr << "Failed to open file" << endl;
        //     return 1;
        // }
        // outfile.write(reinterpret_cast<char*>(model_output), batch_num * 25200 * 85 * sizeof(float));  
        // outfile.close();  

        // delete[] model_output;  


        // std::chrono::steady_clock::time_point starts;
        // starts = std::chrono::steady_clock::now();

        tpu_kernel_launch(ctx->handle, func_id, &api, sizeof(api));
        bm_thread_sync(ctx->handle);

        // std::chrono::steady_clock::time_point ends;
        // ends = std::chrono::steady_clock::now();

        bm_memcpy_d2s_partial_offset(ctx->handle,
                                     (void *)dt_num_data,
                                     detect_num_mem,
                                     batch_num * sizeof(int32_t),
                                     0);
        output_tensor.shape[2] = *dt_num_data;
        if (*dt_num_data != 0) {
            bm_memcpy_d2s_partial_offset(ctx->handle,
                                         (void *)output_data,
                                         out_dev_mem,
                                         output_tensor.shape[2] * output_tensor.shape[3] * sizeof(float),
                                         0);
        }

        // bm_get_profile(ctx->handle, &end);

        // size_t npu_time = end.tpu_process_time - start.tpu_process_time;
        // size_t cdma_time = end.cdma_in_time - start.cdma_in_time + end.cdma_out_time - start.cdma_out_time;
        // std::cout << "npu time = " << npu_time << "(us)" << endl;
        // std::cout << "cdma time = " << cdma_time << "(us)" << endl;
        // std::cout << "time = " << std::chrono::duration_cast<std::chrono::microseconds>(ends - starts).count() << endl;
        return 0;
    }

    std::map< string, std::function<int(int, const TensorVec&,
                                        OutputType&, ContextPtr)>> function_map = 
    {
        { "tpu_kernel_api_yolov5_detect_out", yolov5_decode },
        { "tpu_kernel_api_yolov5_out_without_decode", yolov5_without_decode }
    };

}
#endif
