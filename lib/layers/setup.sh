nvcc -I ~/.local/lib/python3.6/site-packages/megengine/_internal/include -shared -o lib_nms.so -Xcompiler "-fno-strict-aliasing -fPIC" ./gpu_nms/nms.cu
