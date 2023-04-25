import os
import time
import sys
from PIL import Image
import cv2
import glob
from typing import List
import numpy as np
import threading

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import utils.engine as engine_utils
import utils.model as model_utils
import utils.common as common
from utils.common import HostDeviceMem


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
class TextRecognition:
    def __init__(self, trt_engine_path, uff_model_path, trt_engine_datatype=trt.DataType.FLOAT, calib_dataset=None, batch_size=1):
        """Initializes TensorRT objects needed for model inference.

        Args:
            trt_engine_path (str): path where TensorRT engine should be stored
            uff_model_path (str): path of .uff model
            trt_engine_datatype (trt.DataType):
                requested precision of TensorRT engine used for inference
            batch_size (int): batch size for which engine
                should be optimized for
        """
        stream = cuda.Stream()
        trt_runtime = trt.Runtime(TRT_LOGGER)
        trt_engine = None

        # Display requested engine settings to stdout
        print("TensorRT inference engine settings:")
        print("  * Inference precision - {}".format(trt_engine_datatype))
        print("  * Max batch size - {}\n".format(batch_size))

        # If engine is not cached, we need to build it
        if not os.path.exists(trt_engine_path):
            # This function uses supplied .uff file
            # alongside with UffParser to build TensorRT
            # engine. For more details, check implmentation
            trt_engine = engine_utils.build_engine(
                uff_model_path, TRT_LOGGER,
                trt_engine_datatype=trt_engine_datatype,
                calib_dataset=calib_dataset,
                batch_size=batch_size)
            # Save the engine to file
            engine_utils.save_engine(self.trt_engine, trt_engine_path)

        # If we get here, the file with engine exists, so we can load it
        if not trt_engine:
            print("Loading cached TensorRT engine from {}".format(
                trt_engine_path))
            self.engine = engine_utils.load_engine(trt_runtime, trt_engine_path)
            self.context = self.engine.create_execution_context()

        input_names = []
        output_names = []

        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                input_names.append(binding)
            else:
                output_names.append(binding)
  
        self.bindings = [None] * (len(input_names) + len(output_names))

        # store
        self.stream = stream

        self.input_names = input_names
        self.output_names = output_names

        self.input_host_mem = None
        self.input_cuda_mem = None
        self.output_host_mem = None
        self.output_cuda_mem = None

    def destroy(self):
        self.ctx.pop()

    
    def infer_v2(self, image_path):
        """Infers model on given image.

        Args:
            image_path (str): image to run object detection model on
        """
        context = self.context

        stream = self.stream
        engine = self.engine
        bindings = self.bindings

        input_host_mem = self.input_host_mem
        input_cuda_mem = self.input_cuda_mem
        output_host_mem = self.output_host_mem
        output_cuda_mem = self.output_cuda_mem

        # Load image into CPU
        img_src = cv2.imread(image_path)
        img = self._load_img(image_path)

        # Create buffer for inputs and transfer data from CPU to GPU
        idx = engine.get_binding_index(self.input_names[0])
        shape = tuple(img.shape)
        # Create buffer on CPU
        input_host_mem = cuda.pagelocked_empty(shape, np.float32)
        # Create buffer on GPU
        input_cuda_mem = cuda.mem_alloc(input_host_mem.nbytes)
        # Assign to binding
        bindings[idx] = int(input_cuda_mem)
        # Set the dynamic shape for the buffer
        context.set_binding_shape(idx, shape)

        # Create buffer for outputs
        idx = engine.get_binding_index(self.output_names[0])
        dtype = trt.nptype(engine.get_binding_dtype(idx))
        # Get dynamic shape
        shape = tuple(context.get_binding_shape(idx))
        output_host_mem = cuda.pagelocked_empty(shape, np.float32)
        output_cuda_mem = cuda.mem_alloc(output_host_mem.nbytes)
        bindings[idx] = int(output_cuda_mem)

        inference_start_time = time.time()
        np.copyto(input_host_mem, img)
        # Transfer input data to the GPU
        cuda.memcpy_htod_async(input_cuda_mem, input_host_mem, stream)
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU
        cuda.memcpy_dtoh_async(output_host_mem, output_cuda_mem, stream)
        stream.synchronize()

        result = output_host_mem

        print("TensorRT inference time: {} ms".format(
            int(round((time.time() - inference_start_time) * 1000))))
        return result

    def _load_img(self, image_path):
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        if h < 64:
          w = int((64 / h) * w)
        else:
          w = int((h / 64) * w)
        img = cv2.resize(img, (w, 64))[:, :, 0:1]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = img / 255
        img = np.array(img, dtype=float)
        return np.array([img] * 4)

    def preprocess(self):
        pass


if __name__ == "__main__":
    trt_engine_path = "recognizer_trt.engine"
    text_recognizer = TextRecognition(trt_engine_path, "")
    img_paths = glob.glob("line_text_data/*")

    for p in img_paths:
        text_recognizer.infer_v2(p)

      

