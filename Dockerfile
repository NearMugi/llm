FROM ubuntu:latest

RUN apt update && apt install -y g++ gcc make git 

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y tzdata
ENV TZ=Asia/Tokyo

WORKDIR /root
RUN git clone https://github.com/ggerganov/llama.cpp.git

WORKDIR /root/llama.cpp
RUN make -j