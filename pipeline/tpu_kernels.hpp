#ifndef TPUKERNEL_H
#define TPUKERNEL_H

#include <map>
#include <memory>
#include "bmruntime_interface.h"
#include "SGDevicePool.h"
#include "SGLog.h"
#include "bmlib_runtime.h"

using namespace std;

namespace bm {

    /**
     * @todo: add config interface(mask, thresholds, anchors ...)
    */
    int yolov5(int outNum, const TensorVec &outTensors, OutputType &postOut, ContextPtr ctx)
    {
        tpu_kernel_function_t func_id;
        func_id = ctx->getKernelFuncId();

        // bm_profile_t start, end;
        // memset(&start, 0, sizeof(start));
        // memset(&end, 0, sizeof(end));
        // bm_get_profile(ctx->handle, &start);

        int out_len_max = 200 * 7;
        int batch_num = ctx->batchSize;
        float mask[9] = {6, 7, 8, 3, 4, 5, 0, 1, 2};
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
        detect_num.shape[0] = 1;
        detect_num.shape[1] = 1;
        detect_num.shape[2] = 1;
        detect_num.shape[3] = 1;
        detect_num.dims = 4;
        detect_num.data = reinterpret_cast<uint8_t *>(dt_num_data);

        bm_status_t ret = BM_SUCCESS;

        ret = bm_malloc_device_byte(ctx->handle, &out_dev_mem, out_len_max * sizeof(float));
        assert(BM_SUCCESS == ret);
        ret = bm_malloc_device_byte(ctx->handle, &detect_num_mem, batch_num * sizeof(int32_t));
        assert(BM_SUCCESS == ret);

        // config
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
        api.mask_group_size = 3;
        api.keep_top_k = keep_top_k;
        api.nms_threshold = nms_threshold;
        api.confidence_threshold = confidence_threshold;
        memcpy((void *)api.bias, bias, 18 * sizeof(float));
        for (int i = 0; i < outNum; ++i)
        {
            api.anchor_scale[i] = downsample[i];
        }
        memcpy((void *)api.mask, mask, 9 * sizeof(float));
        api.clip_box = 1;

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
        bm_memcpy_d2s_partial_offset(ctx->handle,
                                     (void *)output_data,
                                     out_dev_mem,
                                     output_tensor.shape[2] * output_tensor.shape[3] * sizeof(float),
                                     0);

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
        { "tpu_kernel_api_yolov5_detect_out", yolov5 }
    };

}
#endif
