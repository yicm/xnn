![Build Action](https://github.com/yicm/xnn/actions/workflows/cmake.yml/badge.svg)

# Prerequisites

- Linux-like System.
- CMake 3.1+ installed.
- Use a C++11 compiler(C++11 is optional).
- For benchmark tests
    - OpenCV：`sudo apt install libopencv-dev`

# Build & Install

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ make install DESTDIR={your destination}
```

# Cross-compiling & Install

```bash
$ mkdir build
$ cd build
$ cmake .. -DBUILD_BENCHMARKS=OFF -DCMAKE_TOOLCHAIN_FILE=../platforms/linux/mips32r2-linux-gnu.toolchain.cmake
$ make -j
$ make install DESTDIR={your destination}
```

# Test

```bash
$ cd build/bin/tests
$ cp ../../../conf/config.json .
$ cp ../../../resource/4.gray .
$ ./test_mnn_clazz 4.gray
```

# Benchmarks

```bash
$ cd build/bin/tests
$ cp ../../../conf/config.json .
$ ./test_mnn_clazz_benchmark {your_test_dataset}
```

# Reference

- https://github.com/open-source-parsers/jsoncpp
- https://github.com/gabime/spdlog
- https://github.com/nothings/stb