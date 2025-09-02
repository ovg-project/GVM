#ifndef __UVM_DEBUGFS_API_H__
#define __UVM_DEBUGFS_API_H__

#include "uvm_va_space.h"
#include "uvm_common.h"

int uvm_debugfs_api_preempt_task(uvm_va_space_t *va_space, uvm_gpu_id_t gpu_id);
int uvm_debugfs_api_reschedule_task(uvm_va_space_t *va_space, uvm_gpu_id_t gpu_id);
int uvm_debugfs_api_set_timeslice(uvm_va_space_t *va_space, uvm_gpu_id_t gpu_id, size_t timeslice);
int uvm_debugfs_api_make_realtime(uvm_va_space_t *va_space, uvm_gpu_id_t gpu_id, NvBool realtime);
int uvm_debugfs_api_set_interleave_level(uvm_va_space_t *va_space, uvm_gpu_id_t gpu_id, size_t interleave_level);

#endif // __UVM_DEBUGFS_API_H