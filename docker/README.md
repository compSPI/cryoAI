# Container management

To speed up the process of updating images, the final image is built in 3 steps:
- cryoai-cuda-base: cuda dependencies on top of the original pytorch image
- cryoai-cmake-base: cmake dependencies on top of the cuda-base image
- cryoai-env-base: python dependencies on top of the cmake-base image

GitHub workflows have been setup so the shallowest image is updated based on what file changes were committed.