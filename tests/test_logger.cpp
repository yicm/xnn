#include "logger/logger.hpp"

int main()
{
    HelloLoggerInit(true, true, true, true);

    XNN_LOGGER_TRACE("hello logger, {}", 2020);
    XNN_LOGGER_DEBUG("hello logger, {}", 2020);
    XNN_LOGGER_INFO("hello logger, {}", 2020);
    XNN_LOGGER_WARN("hello logger, {}", 2020);
    XNN_LOGGER_ERROR("hello logger, {}", 2020);
    XNN_LOGGER_CRITICAL("hello logger, {}", 2020);

    HelloLoggerDrop();

    return 0;
}
