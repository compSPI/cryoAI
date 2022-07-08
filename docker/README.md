# Container management

To speed up the process of updating images, the final image is built in 3 steps:
- cryonettorch-cuda-base: cuda dependencies on top of the original pytorch image (see below)
- cryonettorch-cmake-base: cmake dependencies on top of the cuda-base image (see below)
- cryoai: python dependencies on top of the cmake-base image

At the moment, only the last image can be updated. Methods to update the base images will be provided shortly...