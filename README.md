# Prerequisites

- CMake 3.1+ installed.
- Use a C++11 compiler(C++11 is optional).

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
# cmake .. -DCMAKE_TOOLCHAIN_FILE=../platforms/linux/mips32r2-linux-gnu.toolchain.cmake
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

# Reference

- https://github.com/open-source-parsers/jsoncpp
- https://github.com/gabime/spdlog