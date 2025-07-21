#ifndef __GVMDRV_LOG_H__
#define __GVMDRV_LOG_H__

#ifdef __cplusplus
extern "C" {
#endif

// Log levels
enum LogLevel {
  LOG_DEBUG = 0,
  LOG_INFO = 1,
  LOG_WARN = 2,
  LOG_ERROR = 3,
  LOG_FATAL = 4
};

// Logging function declarations
void gvmdrv_log(LogLevel level, const char *file, int line, const char *func,
                const char *message);
void gvmdrv_set_log_level(LogLevel level);
LogLevel gvmdrv_get_log_level(void);

#ifdef __cplusplus
}
#endif

// Simple C++ logging macros
#ifdef __cplusplus

#define LOG_DEBUG(msg) gvmdrv_log(LOG_DEBUG, __FILE__, __LINE__, __func__, msg)
#define LOG_INFO(msg)  gvmdrv_log(LOG_INFO, __FILE__, __LINE__, __func__, msg)
#define LOG_WARN(msg)  gvmdrv_log(LOG_WARN, __FILE__, __LINE__, __func__, msg)
#define LOG_ERROR(msg) gvmdrv_log(LOG_ERROR, __FILE__, __LINE__, __func__, msg)
#define LOG_FATAL(msg) gvmdrv_log(LOG_FATAL, __FILE__, __LINE__, __func__, msg)

#endif // __cplusplus

#endif // __GVMDRV_LOG_H__