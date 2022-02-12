workspace(name = "bazel_mixed_language")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_mixed_language//toolchains/cpp:register.bzl", "my_register_toolchains")

# ----------------------------------------------------------------------
# environment get/set
# ----------------------------------------------------------------------
load("//bazels:my_repository_rules.bzl", "my_set_env_vars")
my_set_env_vars(
    name = "my_set_env_vars"
)
load("@my_set_env_vars//:env.bzl", "HOME")

# ----------------------------------------------------------------------
# bazel_skylib
# ----------------------------------------------------------------------
http_archive(
    name = "bazel_skylib",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
)
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()

# ----------------------------------------------------------------------
# bazelbuild/platforms
# ----------------------------------------------------------------------
http_archive(
    name = "platforms",
    strip_prefix = "platforms-681f1ee032566aa2d443cf0335d012925d9c58d4",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/archive/681f1ee032566aa2d443cf0335d012925d9c58d4.zip",
        "https://github.com/bazelbuild/platforms/archive/681f1ee032566aa2d443cf0335d012925d9c58d4.zip",
    ],
    # shasum -a 256 xx.zip
    sha256 = "ae95e4bfcd9f66e9dc73a92cee0107fede74163f788e3deefe00f3aaae75c431",
)

# ----------------------------------------------------------------------
# register toolchain
# ----------------------------------------------------------------------
my_register_toolchains()

# ---------------------------------------------------------
# Maven dependencies
# ---------------------------------------------------------
RULES_JVM_EXTERNAL_TAG = "3.3"
RULES_JVM_EXTERNAL_SHA = "d85951a92c0908c80bd8551002d66cb23c3434409c814179c0ff026b53544dab"

http_archive(
    name = "rules_jvm_external",
    strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
    sha256 = RULES_JVM_EXTERNAL_SHA,
    url = "https://github.com/bazelbuild/rules_jvm_external/archive/%s.zip" % RULES_JVM_EXTERNAL_TAG,
)

load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    artifacts = [
        "junit:junit:4.12",
        "androidx.appcompat:appcompat:1.2.0",
        "androidx.constraintlayout:constraintlayout:2.0.1",
    ],
    repositories = [
        # Private repositories are supported through HTTP Basic auth
        "http://username:password@localhost:8081/artifactory/my-repository",
        "https://jcenter.bintray.com/",
        "https://maven.google.com",
        "https://repo1.maven.org/maven2",
    ],
)

# ---------------------------------------------------------
# Android
# ---------------------------------------------------------
android_sdk_repository(
    name = "androidsdk",
    path = HOME + "/Android",
    #api_level = 28,
    #build_tools_version = "29.0.2",
)

android_ndk_repository(
    name = "androidndk",
    path = HOME + "/Android/ndk/18.1.5063045",
)
register_toolchains("@androidndk//:all")

# ----------------------------------------------------------------------
# Remote caching & building :Buildfarm
# ----------------------------------------------------------------------
# sha256 sum xx.zip
BUILDFARM_EXTERNAL_COMMIT = "f0cb2c3cd3531cacd828acddc1046e3c6f6cc7fd"
BUILDFARM_EXTERNAL_SHA256 = "7fa105eb4fbaecd7e456af238f716f9c802c143e9627d6fb1c97564f40977f9c"

http_archive(
    name = "build_buildfarm",
    strip_prefix = "bazel-buildfarm-%s" % BUILDFARM_EXTERNAL_COMMIT,
    sha256 = BUILDFARM_EXTERNAL_SHA256,
    url = "https://github.com/bazelbuild/bazel-buildfarm/archive/%s.zip" % BUILDFARM_EXTERNAL_COMMIT,
)

load("@build_buildfarm//:deps.bzl", "buildfarm_dependencies")

buildfarm_dependencies()

load("@build_buildfarm//:defs.bzl", "buildfarm_init")

buildfarm_init()


# ----------------------------------------------------------------------
# Logger
# ----------------------------------------------------------------------
# download spdlog
http_archive(
    name = "spdlog",
    build_file = "//bazels:spdlog.BUILD",
    sha256 = "1e68e9b40cf63bb022a4b18cdc1c9d88eb5d97e4fd64fa981950a9cacf57a4bf",
    strip_prefix = "spdlog-1.8.0",
    urls = ["https://github.com/gabime/spdlog/archive/v1.8.0.tar.gz"],
)

