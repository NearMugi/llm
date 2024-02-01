FROM ubuntu:latest

RUN apt update && apt install -y g++ gcc make vim git

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y tzdata
ENV TZ=Asia/Tokyo

# git clone & build llama 
WORKDIR /root
RUN git clone https://github.com/ggerganov/llama.cpp.git

WORKDIR /root/llama.cpp
RUN make -j

# install python package
RUN apt install pip -y
RUN pip install --upgrade pip
WORKDIR /root
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt
