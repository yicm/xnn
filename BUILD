filegroup(
    name = "includes",
    srcs = glob([
        "3rdparty/target/android_arm64-v8a/include/ncnn/*.h",
    ]),
)

cc_library(
    name = "ncnn",
    hdrs = [":includes"],
    srcs = [
        "3rdparty/target/android_arm64-v8a/lib/libncnn.a",
    ],
    includes = [
        "3rdparty/target/android_arm64-v8a/include/",
    ],
    linkstatic = True,
    visibility = ["//visibility:public"],
)