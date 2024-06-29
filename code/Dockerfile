FROM gcc:14.1.0

RUN apt-get update \
  && apt-get install --no-install-recommends -y \
    cmake \
    gdb \
    libbenchmark-dev \
    valgrind \
  && rm -rf /var/lib/apt/lists/*
  
COPY . /usr/src/project

WORKDIR /usr/src/project/build

RUN cmake .. && make VERBOSE=1
