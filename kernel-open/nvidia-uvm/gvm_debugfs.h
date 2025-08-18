#ifndef _GVM_DEBUGFS_H
#define _GVM_DEBUGFS_H

#include <linux/debugfs.h>
#include <linux/sched.h>

#include "uvm_types.h"
#include "uvm_processors.h"
#include "uvm_va_space.h"

//
// GVM Debugfs interface for GPU process control
//
// This provides a per-process debugfs interface that allows cross-process
// GPU resource management without requiring kernel cgroup modifications.
//

#define GVM_MAX_PROCESSORS UVM_MAX_PROCESSORS

// Per-GPU debugfs directory structure
struct gvm_gpu_debugfs {
    struct dentry *gpu_dir;          // /sys/kernel/debug/nvidia-uvm/processes/<pid>/<gpu_id>/
    struct dentry *memory_limit;      // memory.limit file
    struct dentry *memory_recommend;  // memory.recommend file (read-only)
    struct dentry *memory_current;   // memory.current file (read-only)
    struct dentry *compute_priority;     // compute.priority file
    struct dentry *compute_freeze;       // compute.freeze file
    struct dentry *compute_current;  // compute.current file (read-only)
    pid_t pid;                       // Process ID
    uvm_gpu_id_t gpu_id;                      // GPU ID
};

// Per-process debugfs directory structure
struct gvm_process_debugfs {
    struct hlist_node hash_node;  // Hash table linkage
    struct dentry *process_dir;   // /sys/kernel/debug/nvidia-uvm/processes/<pid>/
    struct gvm_gpu_debugfs gpus[GVM_MAX_PROCESSORS];  // Per-GPU subdirectories
    pid_t pid;                                        // Process ID
    int num_gpus_created;                             // Number of GPU directories created
};

// Main debugfs interface functions
int gvm_debugfs_init(void);
void gvm_debugfs_exit(void);

// Per-process debugfs management
int gvm_debugfs_create_process_dir(pid_t pid);
void gvm_debugfs_remove_process_dir(pid_t pid);
int gvm_debugfs_create_gpu_dir(pid_t pid, uvm_gpu_id_t gpu_id);
int gvm_debugfs_remove_gpu_dir(pid_t pid, uvm_gpu_id_t gpu_id);

// Process tracking functions (needed by debugfs)
struct gpu_process_entry *gvm_find_gpu_process(pid_t pid, bool create);
void gvm_update_gpu_memory_current(pid_t pid, size_t new_current);
void gvm_set_gpu_memory_limit(pid_t pid, size_t limit);

// Exported functions for cross-process control
int gvm_linux_api_get_task_uvmfd(struct task_struct *task, int *uvmfds, size_t size);
int gvm_linux_api_preempt_task(struct task_struct *task, int fd);
int gvm_linux_api_reschedule_task(struct task_struct *task, int fd);
size_t gvm_linux_api_get_gpu_rss(struct task_struct *task, int fd);

int try_charge_gpu_memcg_debugfs(uvm_va_space_t *va_space, uvm_gpu_id_t gpu_id, size_t size);
int try_uncharge_gpu_memcg_debugfs(uvm_va_space_t *va_space, uvm_gpu_id_t gpu_id, size_t size);

size_t get_gpu_memcg_current(uvm_va_space_t *va_space, uvm_gpu_id_t gpu_id);
size_t get_gpu_memcg_limit(uvm_va_space_t *va_space, uvm_gpu_id_t gpu_id);

size_t sum_gpu_memcg_current_all(uvm_gpu_id_t gpu_id);
void calculate_gpu_memcg_recommend_all(uvm_gpu_id_t gpu_id);
void signal_gpu_memcg_current_over_recommend_all(uvm_gpu_id_t gpu_id);
#endif  // _GVM_DEBUGFS_H
