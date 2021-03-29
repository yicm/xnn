#pragma once

// for cmake
#define XNN_VER_MAJOR 0
#define XNN_VER_MINOR 1
#define XNN_VER_PATCH 0

#define XNN_VERSION (XNN_VER_MAJOR * 10000 + XNN_VER_MINOR * 100 + XNN_VER_PATCH)

// for source code
#define _XNN_STR(s) #s
#define XNN_PROJECT_VERSION(major, minor, patch) "v" _XNN_STR(major.minor.patch)
