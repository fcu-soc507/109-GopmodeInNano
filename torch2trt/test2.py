import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
def build_engine(onnx_path, shape = [1,3,64,32]):

   """
   This is the function to create the TensorRT engine
   Args:
      onnx_path : Path to onnx_file. 
      shape : Shape of the input of the ONNX file. 
  """
   explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

   with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
       builder.max_workspace_size = (256 << 20)
       builder.max_batch_size = 1
       with open(onnx_path, 'rb') as model:
           a = parser.parse(model.read())
           print(a)
           print(parser.get_error(0))
       print(network)
       builder.fp16_mode = True
       network.get_input(0).shape = shape
       engine = builder.build_cuda_engine(network)
       return engine

def save_engine(engine, file_name):
   buf = engine.serialize()
   with open(file_name, 'wb') as f:
       f.write(buf)
def load_engine(trt_runtime, plan_path):
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine

#onnx_path = "/home/soc507/deep_sort_pytorch/deep_sort/deep/ournew.onnx"
onnx_path = "/home/soc507/deep_sort_pytorch/deep_sort/deep/oursmalltest.onnx"
eng = build_engine(onnx_path)
save_engine(eng,"smallmodeltest.trt")

