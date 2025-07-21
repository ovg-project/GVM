/**
 * @file gvmdrv_ioctl.h
 * @brief Low-level IOCTL definitions for the NVIDIA GPU Virtual Memory (UVM)
 * driver with GVM extended functionality.
 */

#ifndef __GVMDRV_IOCTL_H__
#define __GVMDRV_IOCTL_H__

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UVM_IS_INITIALIZED 80
#define UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP 81
#define UVM_CTRL_CMD_OPERATE_CHANNEL 82
#define UVM_SET_GMEMCG 83

#define NVA06C_CTRL_CMD_SET_TIMESLICE                                          \
  (0xa06c0103) /* finn: Evaluated from                                         \
                  "(FINN_KEPLER_CHANNEL_GROUP_A_GPFIFO_INTERFACE_ID << 8) |    \
                  NVA06C_CTRL_SET_TIMESLICE_PARAMS_MESSAGE_ID" */
#define NVA06C_CTRL_CMD_GET_TIMESLICE                                          \
  (0xa06c0104) /* finn: Evaluated from                                         \
                  "(FINN_KEPLER_CHANNEL_GROUP_A_GPFIFO_INTERFACE_ID << 8) |    \
                  NVA06C_CTRL_GET_TIMESLICE_PARAMS_MESSAGE_ID" */
#define NVA06F_CTRL_CMD_STOP_CHANNEL                                           \
  (0xa06f0112) /* finn: Evaluated from                                         \
                  "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) |   \
                  NVA06F_CTRL_STOP_CHANNEL_PARAMS_MESSAGE_ID" */
#define NVA06C_CTRL_CMD_PREEMPT                                                \
  (0xa06c0105) /* finn: Evaluated from                                         \
                  "(FINN_KEPLER_CHANNEL_GROUP_A_GPFIFO_INTERFACE_ID << 8) |    \
                  NVA06C_CTRL_PREEMPT_PARAMS_MESSAGE_ID" */
#define NVA06F_CTRL_CMD_GPFIFO_SCHEDULE                                        \
  (0xa06f0103) /* finn: Evaluated from                                         \
                  "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) |   \
                  NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS_MESSAGE_ID" */
#define NVA06F_CTRL_CMD_RESTART_RUNLIST                                        \
  (0xa06f0111) /* finn: Evaluated from                                         \
                  "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) |   \
                  NVA06F_CTRL_RESTART_RUNLIST_PARAMS_MESSAGE_ID" */
#define NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL                                   \
  (0xa06c0107) /* finn: Evaluated from                                         \
                  "(FINN_KEPLER_CHANNEL_GROUP_A_GPFIFO_INTERFACE_ID << 8) |    \
                  NVA06C_CTRL_SET_INTERLEAVE_LEVEL_PARAMS_MESSAGE_ID" */
#define NVA06F_CTRL_CMD_BIND                                                   \
  (0xa06f0104) /* finn: Evaluated from                                         \
                  "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) |   \
                  NVA06F_CTRL_BIND_PARAMS_MESSAGE_ID" */

typedef struct {
  unsigned long long timesliceUs;
} NVA06C_CTRL_TIMESLICE_PARAMS;

typedef struct {
  int bWait;
  int bManualTimeout;
  unsigned int timeoutUs;
} NVA06C_CTRL_PREEMPT_PARAMS;

typedef struct {
  unsigned int tsgInterleaveLevel; // IN
} NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS;

typedef union {
  NVA06C_CTRL_TIMESLICE_PARAMS NVA06C_CTRL_TIMESLICE_PARAMS;
  NVA06C_CTRL_PREEMPT_PARAMS NVA06C_CTRL_PREEMPT_PARAMS;
  NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS;
} UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS_DATA;

typedef struct {
  bool bEnable;     // IN
  bool bSkipSubmit; // IN
  bool bSkipEnable; // IN
} NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS;

typedef struct {
  bool bForceRestart;
  bool bBypassWait;
} NVA06F_CTRL_RESTART_RUNLIST_PARAMS;

typedef struct {
  bool bImmediate;
} NVA06F_CTRL_STOP_CHANNEL_PARAMS;

typedef struct NVA06F_CTRL_BIND_PARAMS {
  unsigned int engineType;
} NVA06F_CTRL_BIND_PARAMS;

typedef union {
  NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS;
  NVA06F_CTRL_RESTART_RUNLIST_PARAMS NVA06F_CTRL_RESTART_RUNLIST_PARAMS;
  NVA06F_CTRL_STOP_CHANNEL_PARAMS NVA06F_CTRL_STOP_CHANNEL_PARAMS;
  NVA06F_CTRL_BIND_PARAMS NVA06F_CTRL_BIND_PARAMS;
} UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS_DATA;

typedef struct {
  unsigned int cmd;                                    // IN
  UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS_DATA data; // IN
  size_t dataSize;                                     // IN
  int rmStatus;                                        // OUT
} UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS;

typedef struct {
  unsigned int cmd;                              // IN
  UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS_DATA data; // IN
  size_t dataSize;                               // IN
  int rmStatus;                                  // OUT
} UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS;

typedef struct {
  bool initialized; // IN/OUT
  int rmStatus;     // OUT
} UVM_IS_INITIALIZED_PARAMS;

typedef struct {
  size_t size;  // IN
  int rmStatus; // OUT
} UVM_SET_GMEMCG_PARAMS;

#ifdef __cplusplus
}
#endif

#endif // __GVMDRV_IOCTL_H__
