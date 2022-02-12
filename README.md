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

## Build Android JNI Library

Environment：

- WSL,Ubuntu18.04
- Bazel 3.3+
- Android JNI Library
    - Local JDK 8+(just for using `javah` or `javac` to generate native headers)
        - `sudo apt install openjdk-8-jdk-headless`
    - Android sdkmanager (https://developer.android.com/studio#command-tools)
        - config the environment of `sdkmanager`
        - `sdkmanager --sdk_root=$HOME/Android --list`
        - `sdkmanager --sdk_root=$HOME/Android 'build-tools;29.0.2'`
        - `sdkmanager --sdk_root=$HOME/Android 'platforms;android-28'`
        - `sdkmanager --sdk_root=$HOME/Android --install "ndk;18.1.5063045"`

Build:

```shell
$ bazel build --platforms=//platforms:p_android_aarch64 android:jni_lib_shared
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
# Online Tools

- https://rawpixels.net
- https://www.metadata2go.com
