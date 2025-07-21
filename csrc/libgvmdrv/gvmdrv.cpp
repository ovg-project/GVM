#include <errno.h>
#include <string.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>

#include "gvmdrv.h"
#include "gvmdrv_ioctl.h"
#include "gvmdrv_log.h"

namespace gvmdrv {

int find_initialized_uvm() {
  pid_t pid = getpid();
  const std::string target_path = "/dev/nvidia-uvm";

  std::string fd_dir = "/proc/" + std::to_string(pid) + "/fd";
  for (const auto &entry : std::filesystem::directory_iterator(fd_dir)) {
    std::error_code ec;
    std::string fd_path = std::filesystem::read_symlink(entry, ec).string();
    if (ec)
      continue;

    if (fd_path == target_path) {
      int fd = std::stoi(entry.path().filename());
      if (fd >= 0) {
        UVM_IS_INITIALIZED_PARAMS params = {.initialized = false,
                                            .rmStatus = 0};
        if (ioctl(fd, UVM_IS_INITIALIZED, &params) == 0) {
          if (params.initialized) {
            return fd;
          }
        } else {
          LOG_ERROR("Failed to check UVM initialization");
        }
      } else {
        LOG_ERROR("Invalid file descriptor");
      }
    }
  }
  LOG_ERROR("No initialized UVM found");
  return -1;
}

void set_timeslice(int fd, long long unsigned timesliceUs) {
  if (fd < 0) {
    LOG_ERROR("Invalid file descriptor");
    return;
  }

  UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS params = {
      .cmd = NVA06C_CTRL_CMD_SET_TIMESLICE,
      .data = {.NVA06C_CTRL_TIMESLICE_PARAMS = {.timesliceUs = timesliceUs}},
      .dataSize = sizeof(NVA06C_CTRL_TIMESLICE_PARAMS),
      .rmStatus = 0};
  int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP, &params);
  if (ret < 0) {
    LOG_ERROR("Failed to execute ioctl command: SET_TIMESLICE");
  } else {
    LOG_INFO("Timeslice set successfully");
  }
}

long long unsigned get_timeslice(int fd) {
  if (fd < 0) {
    LOG_ERROR("Invalid file descriptor");
    return 0;
  }

  long long unsigned timesliceUs = 0;

  UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS params = {
      .cmd = NVA06C_CTRL_CMD_GET_TIMESLICE,
      .data = {.NVA06C_CTRL_TIMESLICE_PARAMS = {.timesliceUs = timesliceUs}},
      .dataSize = sizeof(NVA06C_CTRL_TIMESLICE_PARAMS),
      .rmStatus = 0};
  int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP, &params);
  if (ret < 0) {
    LOG_ERROR("Failed to execute ioctl command: GET_TIMESLICE");
  }

  return params.data.NVA06C_CTRL_TIMESLICE_PARAMS.timesliceUs;
}

void preempt(int fd) {
  if (fd < 0) {
    LOG_ERROR("Invalid file descriptor");
    return;
  }

  UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS params = {
      .cmd = NVA06C_CTRL_CMD_PREEMPT,
      .data = {.NVA06C_CTRL_PREEMPT_PARAMS = {.bWait = true,
                                              .bManualTimeout = false,
                                              .timeoutUs = 1000}},
      .dataSize = sizeof(NVA06C_CTRL_PREEMPT_PARAMS),
      .rmStatus = 0};
  int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP, &params);
  if (ret < 0) {
    LOG_ERROR("Failed to execute ioctl command: PREEMPT");
  } else {
    LOG_INFO("Preempted successfully");
  }
}

void restart(int fd) {
  if (fd < 0) {
    LOG_ERROR("Invalid file descriptor");
    return;
  }

  UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS params = {
      .cmd = NVA06F_CTRL_CMD_RESTART_RUNLIST,
      .data = {.NVA06F_CTRL_RESTART_RUNLIST_PARAMS = {.bForceRestart = true,
                                                      .bBypassWait = false}},
      .dataSize = sizeof(NVA06F_CTRL_CMD_RESTART_RUNLIST),
      .rmStatus = 0};
  int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL, &params);
  if (ret < 0) {
    LOG_ERROR("Failed to execute ioctl command: RESTART_RUNLIST");
  } else {
    LOG_INFO("Restarted successfully");
  }
}

void schedule(int fd, bool enable) {
  if (fd < 0) {
    LOG_ERROR("Invalid file descriptor");
    return;
  }

  UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS params = {
      .cmd = NVA06F_CTRL_CMD_GPFIFO_SCHEDULE,
      .data = {.NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS = {.bEnable = enable,
                                                      .bSkipSubmit = !enable,
                                                      .bSkipEnable = !enable}},
      .dataSize = sizeof(NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS),
      .rmStatus = 0};
  int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL, &params);
  if (ret < 0) {
    LOG_ERROR("Failed to execute ioctl command: GPFIFO_SCHEDULE");
  } else {
    LOG_INFO("Scheduled successfully");
  }
}

void stop(int fd) {
  if (fd < 0) {
    LOG_ERROR("Invalid file descriptor");
    return;
  }

  UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS params = {
      .cmd = NVA06F_CTRL_CMD_STOP_CHANNEL,
      .data = {.NVA06F_CTRL_STOP_CHANNEL_PARAMS = {.bImmediate = true}},
      .dataSize = sizeof(NVA06F_CTRL_STOP_CHANNEL_PARAMS),
      .rmStatus = 0};
  int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL, &params);
  if (ret < 0) {
    LOG_ERROR("Failed to stop");
  } else {
    LOG_INFO("Stopped successfully");
  }
}

void set_interleave(int fd, uint32_t interleave) {
  if (fd < 0) {
    LOG_ERROR("Invalid file descriptor");
    return;
  }

  UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS params = {
      .cmd = NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL,
      .data = {.NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS = {.tsgInterleaveLevel =
                                                           interleave}},
      .dataSize = sizeof(NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS),
      .rmStatus = 0};
  int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP, &params);
  if (ret < 0) {
    LOG_ERROR("Failed to set interleave level");
  } else {
    LOG_INFO("Set interleave level successfully");
  }
}

void bind(int fd) {
  if (fd < 0) {
    LOG_ERROR("Invalid file descriptor");
    return;
  }

  UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS params = {
      .cmd = NVA06F_CTRL_CMD_BIND,
      .data =
          {.NVA06F_CTRL_BIND_PARAMS =
               {// Will be filled in kernel module using restored type of engine
                .engineType = 0}},
      .dataSize = sizeof(NVA06F_CTRL_BIND_PARAMS),
      .rmStatus = 0};
  int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL, &params);
  if (ret < 0) {
    LOG_ERROR("Failed to bind");
  } else {
    LOG_INFO("Binded successfully");
  }
}

void set_gmemcg(int fd, unsigned long long size) {
  if (fd < 0) {
    LOG_ERROR("Invalid file descriptor");
    return;
  }

  UVM_SET_GMEMCG_PARAMS params = {.size = size, .rmStatus = 0};
  int ret = ioctl(fd, UVM_SET_GMEMCG, &params);
  if (ret < 0 || params.rmStatus != 0) {
    LOG_ERROR("Failed to set gmemcg");
  } else {
    LOG_INFO("Set gmemcg successfully");
  }
}

} // namespace gvmdrv

// C-compatible function definitions
extern "C" {

int gvm_find_initialized_uvm() { return gvmdrv::find_initialized_uvm(); }

void gvm_set_timeslice(int fd, long long unsigned timesliceUs) {
  gvmdrv::set_timeslice(fd, timesliceUs);
}

long long unsigned gvm_get_timeslice(int fd) { return gvmdrv::get_timeslice(fd); }

void gvm_preempt(int fd) { gvmdrv::preempt(fd); }

void gvm_restart(int fd) { gvmdrv::restart(fd); }

void gvm_schedule(int fd, bool enable) { gvmdrv::schedule(fd, enable); }

void gvm_stop(int fd) { gvmdrv::stop(fd); }

void gvm_set_interleave(int fd, unsigned int interleave) {
  gvmdrv::set_interleave(fd, interleave);
}

void gvm_bind(int fd) { gvmdrv::bind(fd); }

void gvm_set_gmemcg(int fd, unsigned long long size) {
  gvmdrv::set_gmemcg(fd, size);
}
} // extern "C"
