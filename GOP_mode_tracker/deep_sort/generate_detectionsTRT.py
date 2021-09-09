# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2
#import tensorflow as tf
import time
import tensorrt as trt 
import pycuda.driver as cuda



class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    img1 = cv2.resize(image, tuple(patch_shape[::-1]))
    img1 = cv2.normalize(img1, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img1 = img1.swapaxes(0,1)
    img1 = img1.swapaxes(0,2)
    img1 = img1[np.newaxis,:]
    return img1
def allocate_buffers(engine):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []
    output_idx = 0
    stream = cuda.Stream()
    #assert 3 <= len(engine) <= 4  # expect 1 input, plus 2 or 3 outpus
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * \
               engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            # each grid has 3 anchors, each anchor generates a detection
            # output of 7 float32 values
            #assert size % 7 == 0
            outputs.append(HostDeviceMem(host_mem, device_mem))
            output_idx += 1
    return inputs, outputs, bindings, stream

def do_inference_v2(context, bindings, inputs, outputs, stream):
    """do_inference_v2 (for TensorRT 7.0+)

    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    t0 = time.time()
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    t1 = time.time()
    
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    t2 = time.time()
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    t3 = time.time()
    stream.synchronize()
    t4 = time.time()
    #print("CTG:",t1-t0)
    #print("INF:",t2-t1)
    #print("GTC:",t3-t2)
    #print("STS:",t4-t3)
    # Return only the host outputs.
    return [out.host for out in outputs],(t4-t3)
#train to TRT
class ImageEncoder(object):

    def _load_engine(self):
        TRTbin = '%s.trt' % self.model
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    def __init__(self, model, input_shape, letter_box=False,
                 cuda_ctx=None):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.input_shape = input_shape
        self.letter_box = letter_box
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()
        self.image_shape = np.array([128,64])
        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = \
                allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()
    #def __init__(self, checkpoint_filename, input_name="images",
    #             output_name="features"):
    
    def __del__(self):
        """Free CUDA memories."""
        del self.outputs
        del self.inputs
        del self.stream


    def __call__(self, data_x, batch_size=1):
        e0 = time.time()
        out = np.zeros((len(data_x), 128), np.float32)
        #_run_in_batches(do_inference_v2,data_x, out, batch_size)
        e1 = time.time()
        #def _run_in_batches(f,data_dict, out, batch_size):
        data_len = len(out)
        num_batches = int(data_len)
        v = data_x
        #td,ta =0,0
        s, e = 0, 0
        for i in range(num_batches):
            d0 = time.time()
            s, e = i , (i + 1)
            self.inputs[0].host = np.ascontiguousarray(v[s:e]) 
            d1 = time.time()
            out[s:e],tim0 = do_inference_v2(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)
            #td+=(d1-d0)
            #ta +=tim0
        if e < len(out):        
            self.inputs[0].host = np.ascontiguousarray(v[e:]) 
            #out[e:] = do_inference_v2(self.inputs)
            out[e:] = do_inference_v2(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)
            #batch_data_dict =  v[e:] 
            #out[e:] = do_inference_v2(batch_data_dict)
        #print("preset",(e1-e0)+td)
        #print("STS",ta/data_len)
        return out

def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=1):
    image_encoder = ImageEncoder(model_filename, [1,3,128,64])
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        #print(np.shape(boxes))
        for box in boxes:
            #print("sir! this is box",box)
            #box[0],box[1],box[2],box[3] = float(box[0]),float(box[1]),float(box[2]),float(box[3])
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        o = image_encoder(image_patches, batch_size) 
        return o

    return encoder


