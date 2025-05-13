##########################################
# Dockerfile for TD-MPC2                 #
# Nicklas Hansen, 2023 (c)               #
# https://www.tdmpc2.com                 #
# -------------------------------------- #
# Build instructions:                    #
# docker build . -t <user>/tdmpc2:1.0.1  #
# docker push <user>/tdmpc2:1.0.1        #
# -------------------------------------- #
# Run:                                   #
# docker run -i \                        #
#   -v <path>/<to>/tdmpc2:/tdmpc2 \      #
#   --gpus all \                         #
#   -t <user>/tdmpc2:1.0.1 \             #
#   /bin/bash                            #
##########################################

# base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive

# packages
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends build-essential git nano rsync vim tree curl wget \
    swig ffmpeg unzip htop tmux xvfb ca-certificates bash-completion libjpeg-dev libpng-dev \
    libssl-dev libcurl4-openssl-dev libopenmpi-dev zlib1g-dev qtbase5-dev qtdeclarative5-dev \
    libglib2.0-0 libglu1-mesa-dev libgl1-mesa-dev libvulkan1 libgl1-mesa-glx libosmesa6 \
    libosmesa6-dev libglew-dev mesa-utils && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /root/.ssh

# conda environment
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY environment.yaml /root
RUN conda update conda && \
    conda env update -n base -f /root/environment.yaml && \
    rm /root/environment.yaml && \
    conda clean -ya && \
    pip cache purge && \
    conda init
SHELL ["/bin/bash", "-c"]
RUN echo "cd /root" >> /root/.bashrc
# image does not include metaworld, maniskill, myosuite
# these can be installed separately; see environment.yaml for details

# success!
RUN echo "Successfully built TD-MPC2 Docker image!"
