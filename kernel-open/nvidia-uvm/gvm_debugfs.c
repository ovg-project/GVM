#include <linux/debugfs.h>
#include <linux/uaccess.h>
#include <linux/seq_file.h>
#include <linux/pid.h>
#include <linux/sched/signal.h>
#include <linux/hashtable.h>
#include <linux/slab.h>

#include "gvm_debugfs.h"
#include "uvm_common.h"
#include "uvm_debugfs_api.h"
#include "uvm_global.h"
#include "uvm_va_block.h"

// Global debugfs root directory
static struct dentry *gvm_debugfs_root;
static struct dentry *gvm_debugfs_processes_dir;

// Hash table for per-process debugfs directories
#define GVM_DEBUGFS_HASH_BITS 8
static DEFINE_HASHTABLE(gvm_debugfs_dirs, GVM_DEBUGFS_HASH_BITS);
static DEFINE_SPINLOCK(gvm_debugfs_lock);

#define GVM_MAX_VA_SPACES 8

// Timeslice is calculated by GVM_MAX_TIMESLICE_US >> priority
// In default, priority is 2, whose timeslice is 2048 us
#define GVM_MIN_PRIORITY 16
#define GVM_MAX_TIMESLICE_US 524288

//
// Forward declarations of util functions
//

static uvm_va_space_t *_gvm_find_va_space_by_pid(pid_t pid);
static uvm_gpu_cgroup_t *_gvm_find_gpu_cgroup_by_pid(pid_t pid);
static size_t _gvm_find_and_acquire_va_spaces_by_pid(pid_t pid, uvm_va_space_t **va_spaces, size_t size);
static size_t _gvm_release_va_spaces(uvm_va_space_t **va_spaces, size_t size);

//
// Per-process debugfs file operations
//

// Show memory limit for a specific process and GPU
static int gvm_process_memory_limit_show(struct seq_file *m, void *data)
{
    struct gvm_gpu_debugfs *gpu_debugfs;
    uvm_gpu_cgroup_t *gpu_cgroup;

    gpu_debugfs = m->private;
    if (!gpu_debugfs)
        return -ENOENT;

    gpu_cgroup = _gvm_find_gpu_cgroup_by_pid(gpu_debugfs->pid);
    if (!gpu_cgroup)
        return -ENOENT;

    seq_printf(m, "%zu\n", gpu_cgroup[uvm_id_gpu_index(gpu_debugfs->gpu_id)].memory_limit);

    return 0;
}

// Set memory limit for a specific process and GPU
static ssize_t gvm_process_memory_limit_write(struct file *file, const char __user *user_buf,
                                             size_t count, loff_t *ppos)
{
    struct seq_file *m = file->private_data;
    struct gvm_gpu_debugfs *gpu_debugfs = m->private;
    uvm_va_space_t *va_spaces[GVM_MAX_VA_SPACES];
    size_t va_space_index;
    size_t va_space_count;
    char buf[32];
    size_t limit;
    int error;

    if (count >= sizeof(buf))
        return -EINVAL;

    if (copy_from_user(buf, user_buf, count))
        return -EFAULT;

    buf[count] = '\0';

    error = kstrtoul(buf, 10, (unsigned long *) &limit);
    if (error != 0)
        return error;

    va_space_count = _gvm_find_and_acquire_va_spaces_by_pid(gpu_debugfs->pid, va_spaces, GVM_MAX_VA_SPACES);
    if (va_space_count == 0)
        return -ENOENT;

    for (va_space_index = 0; va_space_index < va_space_count; ++va_space_index) {
        if (va_spaces[va_space_index]->gpu_cgroup == NULL)
            continue;

        va_spaces[va_space_index]->gpu_cgroup[uvm_id_gpu_index(gpu_debugfs->gpu_id)].memory_limit = limit;
        uvm_debugfs_api_charge_gpu_memory_limit(va_spaces[va_space_index], gpu_debugfs->gpu_id, va_spaces[va_space_index]->gpu_cgroup[uvm_id_gpu_index(gpu_debugfs->gpu_id)].memory_current, limit);
    }

    _gvm_release_va_spaces(va_spaces, va_space_count);
    return count;
}

// Show current memory usage for a specific process and GPU
static int gvm_process_memory_current_show(struct seq_file *m, void *data)
{
    struct gvm_gpu_debugfs *gpu_debugfs;
    uvm_gpu_cgroup_t *gpu_cgroup;

    gpu_debugfs = m->private;
    if (!gpu_debugfs)
        return -ENOENT;

    gpu_cgroup = _gvm_find_gpu_cgroup_by_pid(gpu_debugfs->pid);
    if (!gpu_cgroup)
        return -ENOENT;

    seq_printf(m, "%zu\n", gpu_cgroup[uvm_id_gpu_index(gpu_debugfs->gpu_id)].memory_current);

    return 0;
}

// Show current swap memory usage for a specific process and GPU
static int gvm_process_memory_swap_current_show(struct seq_file *m, void *data)
{
    struct gvm_gpu_debugfs *gpu_debugfs;
    uvm_gpu_cgroup_t *gpu_cgroup;

    gpu_debugfs = m->private;
    if (!gpu_debugfs)
        return -ENOENT;

    gpu_cgroup = _gvm_find_gpu_cgroup_by_pid(gpu_debugfs->pid);
    if (!gpu_cgroup)
        return -ENOENT;

    seq_printf(m, "%zu\n", gpu_cgroup[uvm_id_gpu_index(gpu_debugfs->gpu_id)].memory_swap_current);

    return 0;
}

// Show current compute timeslice for a specific process and GPU
static int gvm_process_compute_priority_show(struct seq_file *m, void *data)
{
    struct gvm_gpu_debugfs *gpu_debugfs;
    uvm_gpu_cgroup_t *gpu_cgroup;

    gpu_debugfs = m->private;
    if (!gpu_debugfs)
        return -ENOENT;

    gpu_cgroup = _gvm_find_gpu_cgroup_by_pid(gpu_debugfs->pid);
    if (!gpu_cgroup)
        return -ENOENT;

    seq_printf(m, "%zu\n", gpu_cgroup[uvm_id_gpu_index(gpu_debugfs->gpu_id)].compute_priority);

    return 0;
}

// Set compute timeslice limit for a specific process and GPU
static ssize_t gvm_process_compute_priority_write(struct file *file, const char __user *user_buf,
                                              size_t count, loff_t *ppos)
{
    struct seq_file *m = file->private_data;
    struct gvm_gpu_debugfs *gpu_debugfs = m->private;
    int error = 0;
    uvm_va_space_t *va_spaces[GVM_MAX_VA_SPACES];
    size_t va_space_index;
    size_t va_space_count;
    char buf[32];
    size_t priority;

    if (count >= sizeof(buf))
        return -EINVAL;

    if (copy_from_user(buf, user_buf, count))
        return -EFAULT;

    buf[count] = '\0';

    error = kstrtoul(buf, 10, (unsigned long *) &priority);
    if (error != 0)
        return error;

    if (priority > GVM_MIN_PRIORITY) {
        UVM_ERR_PRINT("priority should range from 0 to %d but got %lu\n", GVM_MIN_PRIORITY, priority);
        return -EINVAL;
    }

    va_space_count = _gvm_find_and_acquire_va_spaces_by_pid(gpu_debugfs->pid, va_spaces, GVM_MAX_VA_SPACES);
    if (va_space_count == 0)
        return -ENOENT;

    for (va_space_index = 0; va_space_index < va_space_count; ++va_space_index) {
        if (va_spaces[va_space_index]->gpu_cgroup == NULL)
            continue;

        va_spaces[va_space_index]->gpu_cgroup[uvm_id_gpu_index(gpu_debugfs->gpu_id)].compute_priority = priority;

        error = uvm_debugfs_api_set_timeslice(va_spaces[va_space_index], gpu_debugfs->gpu_id, GVM_MAX_TIMESLICE_US >> priority);
        if (error)
            break;
    }

    _gvm_release_va_spaces(va_spaces, va_space_count);
    return error ? error : count;
}

// Show current compute freeze status for a specific process and GPU
static int gvm_process_compute_freeze_show(struct seq_file *m, void *data)
{
    struct gvm_gpu_debugfs *gpu_debugfs;
    uvm_gpu_cgroup_t *gpu_cgroup;

    gpu_debugfs = m->private;
    if (!gpu_debugfs)
        return -ENOENT;

    gpu_cgroup = _gvm_find_gpu_cgroup_by_pid(gpu_debugfs->pid);
    if (!gpu_cgroup)
        return -ENOENT;

    seq_printf(m, "%zu\n", gpu_cgroup[uvm_id_gpu_index(gpu_debugfs->gpu_id)].compute_freeze);

    return 0;
}

// Set compute freeze status and preempt/reschedule for a specific process and GPU
static ssize_t gvm_process_compute_freeze_write(struct file *file, const char __user *user_buf,
                                              size_t count, loff_t *ppos)
{
    struct seq_file *m = file->private_data;
    struct gvm_gpu_debugfs *gpu_debugfs = m->private;
    int error = 0;
    uvm_va_space_t *va_spaces[GVM_MAX_VA_SPACES];
    size_t va_space_index;
    size_t va_space_count;
    char buf[32];
    size_t freeze;

    if (count >= sizeof(buf))
        return -EINVAL;

    if (copy_from_user(buf, user_buf, count))
        return -EFAULT;

    buf[count] = '\0';

    error = kstrtoul(buf, 10, (unsigned long *) &freeze);
    if (error != 0)
        return error;

    if (freeze > 1) {
        UVM_ERR_PRINT("freeze should be 0 or 1 but got %lu\n", freeze);
        return -EINVAL;
    }

    va_space_count = _gvm_find_and_acquire_va_spaces_by_pid(gpu_debugfs->pid, va_spaces, GVM_MAX_VA_SPACES);
    if (va_space_count == 0)
        return -ENOENT;

    for (va_space_index = 0; va_space_index < va_space_count; ++va_space_index) {
        if (va_spaces[va_space_index]->gpu_cgroup == NULL)
            continue;

        va_spaces[va_space_index]->gpu_cgroup[uvm_id_gpu_index(gpu_debugfs->gpu_id)].compute_freeze = freeze;

        error = uvm_debugfs_api_schedule_task(va_spaces[va_space_index], gpu_debugfs->gpu_id, freeze);
        if (error)
            break;
    }

    _gvm_release_va_spaces(va_spaces, va_space_count);
    return error ? error : count;
}

static int gvm_process_gcgroup_stat_show(struct seq_file *m, void *data)
{
    struct gvm_gpu_debugfs *gpu_debugfs;
    uvm_gpu_cgroup_t *gpu_cgroup;
    size_t nr_submitted_kernels;
    size_t nr_ended_kernels;

    gpu_debugfs = m->private;
    if (!gpu_debugfs)
        return -ENOENT;

    gpu_cgroup = _gvm_find_gpu_cgroup_by_pid(gpu_debugfs->pid);
    if (!gpu_cgroup)
        return -ENOENT;

    nr_submitted_kernels = (size_t)atomic64_read(&(gpu_cgroup[uvm_id_gpu_index(gpu_debugfs->gpu_id)].nr_submitted_kernels));
    nr_ended_kernels = (size_t)atomic64_read(&(gpu_cgroup[uvm_id_gpu_index(gpu_debugfs->gpu_id)].nr_ended_kernels));
    seq_printf(m, "nr_submitted_kernels: %zu\nnr_ended_kernels: %zu\nnr_pending_kernels: %zu\n",
            nr_submitted_kernels, nr_ended_kernels,
            (nr_submitted_kernels > nr_ended_kernels) ? nr_submitted_kernels - nr_ended_kernels : 0);

    return 0;
}

//
// File operation structures
//

static int gvm_process_memory_limit_open(struct inode *inode, struct file *file)
{
    return single_open(file, gvm_process_memory_limit_show, inode->i_private);
}

static const struct file_operations gvm_process_memory_limit_fops = {
    .open = gvm_process_memory_limit_open,
    .read = seq_read,
    .write = gvm_process_memory_limit_write,
    .llseek = seq_lseek,
    .release = single_release,
};

static int gvm_process_memory_current_open(struct inode *inode, struct file *file)
{
    return single_open(file, gvm_process_memory_current_show, inode->i_private);
}

static const struct file_operations gvm_process_memory_current_fops = {
    .open = gvm_process_memory_current_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = single_release,
};

static int gvm_process_memory_swap_current_open(struct inode *inode, struct file *file)
{
    return single_open(file, gvm_process_memory_swap_current_show, inode->i_private);
}

static const struct file_operations gvm_process_memory_swap_current_fops = {
    .open = gvm_process_memory_swap_current_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = single_release,
};

static int gvm_process_compute_priority_open(struct inode *inode, struct file *file)
{
    return single_open(file, gvm_process_compute_priority_show, inode->i_private);
}

static const struct file_operations gvm_process_compute_priority_fops = {
    .open = gvm_process_compute_priority_open,
    .read = seq_read,
    .write = gvm_process_compute_priority_write,
    .llseek = seq_lseek,
    .release = single_release,
};

static int gvm_process_compute_freeze_open(struct inode *inode, struct file *file)
{
    return single_open(file, gvm_process_compute_freeze_show, inode->i_private);
}

static const struct file_operations gvm_process_compute_freeze_fops = {
    .open = gvm_process_compute_freeze_open,
    .read = seq_read,
    .write = gvm_process_compute_freeze_write,
    .llseek = seq_lseek,
    .release = single_release,
};

static int gvm_process_gcgroup_stat_open(struct inode *inode, struct file *file)
{
    return single_open(file, gvm_process_gcgroup_stat_show, inode->i_private);
}

static const struct file_operations gvm_process_gcgroup_stat_fops = {
    .open = gvm_process_gcgroup_stat_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = single_release,
};

//
// Global process list (for debugging)
//

static int gvm_processes_list_show(struct seq_file *m, void *data)
{
    struct gvm_process_debugfs *proc_debugfs;
    struct hlist_node *tmp;
    int bucket;

    seq_printf(m, "PID\n");
    spin_lock(&gvm_debugfs_lock);
    hash_for_each_safe(gvm_debugfs_dirs, bucket, tmp, proc_debugfs, hash_node)
    {
        seq_printf(m, "%d\n", proc_debugfs->pid);
    }
    spin_unlock(&gvm_debugfs_lock);
    return 0;
}

static int gvm_processes_list_open(struct inode *inode, struct file *file)
{
    return single_open(file, gvm_processes_list_show, NULL);
}

static const struct file_operations gvm_processes_list_fops = {
    .open = gvm_processes_list_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = single_release,
};

//
// Per-process directory management
//

int gvm_debugfs_create_process_dir(pid_t pid)
{
    struct gvm_process_debugfs *proc_debugfs;
    char process_dirname[16];
    int ret = 0;

    // Check if directory already exists
    spin_lock(&gvm_debugfs_lock);
    hash_for_each_possible(gvm_debugfs_dirs, proc_debugfs, hash_node, pid)
    {
        if (proc_debugfs->pid == pid) {
            spin_unlock(&gvm_debugfs_lock);
            return 0;  // Already exists
        }
    }
    spin_unlock(&gvm_debugfs_lock);

    // Allocate new debugfs directory structure
    proc_debugfs = kzalloc(sizeof(*proc_debugfs), GFP_KERNEL);
    if (!proc_debugfs)
        return -ENOMEM;

    proc_debugfs->pid = pid;
    proc_debugfs->num_gpus_created = 0;

    // Create process directory
    snprintf(process_dirname, sizeof(process_dirname), "%d", pid);
    proc_debugfs->process_dir = debugfs_create_dir(process_dirname, gvm_debugfs_processes_dir);
    if (!proc_debugfs->process_dir) {
        ret = -ENOMEM;
        goto cleanup_proc;
    }

    // GPU subdirectories will be created lazily when GPUs are registered with UVM
    // This avoids the issue of not knowing how many GPUs are available at process creation time

    // Add to hash table
    spin_lock(&gvm_debugfs_lock);
    hash_add(gvm_debugfs_dirs, &proc_debugfs->hash_node, pid);
    spin_unlock(&gvm_debugfs_lock);

    return 0;

cleanup_proc:
    kfree(proc_debugfs);
    return ret;
}

void gvm_debugfs_remove_process_dir(pid_t pid)
{
    struct gvm_process_debugfs *proc_debugfs;

    spin_lock(&gvm_debugfs_lock);
    hash_for_each_possible(gvm_debugfs_dirs, proc_debugfs, hash_node, pid)
    {
        if (proc_debugfs->pid == pid) {
            hash_del(&proc_debugfs->hash_node);
            spin_unlock(&gvm_debugfs_lock);

            // Remove all GPU directories and files recursively
            debugfs_remove_recursive(proc_debugfs->process_dir);
            kfree(proc_debugfs);
            return;
        }
    }
    spin_unlock(&gvm_debugfs_lock);
}

int gvm_debugfs_create_gpu_dir(pid_t pid, uvm_gpu_id_t gpu_id)
{
    uvm_va_space_t *va_space = _gvm_find_va_space_by_pid(pid);
    struct gvm_process_debugfs *proc_debugfs = NULL;
    struct gvm_gpu_debugfs *gpu_debugfs;
    char gpu_dirname[16];
    int ret = 0;

    // Find the process debugfs entry
    spin_lock(&gvm_debugfs_lock);
    hash_for_each_possible(gvm_debugfs_dirs, proc_debugfs, hash_node, pid)
    {
        if (proc_debugfs->pid == pid) {
            break;
        }
        proc_debugfs = NULL;
    }
    spin_unlock(&gvm_debugfs_lock);

    if (!proc_debugfs) {
        // Process directory doesn't exist, create it first
        ret = gvm_debugfs_create_process_dir(pid);
        if (ret != 0)
            return ret;

        // Try to find it again
        spin_lock(&gvm_debugfs_lock);
        hash_for_each_possible(gvm_debugfs_dirs, proc_debugfs, hash_node, pid)
        {
            if (proc_debugfs->pid == pid) {
                break;
            }
            proc_debugfs = NULL;
        }
        spin_unlock(&gvm_debugfs_lock);

        if (!proc_debugfs)
            return -ENOENT;
    }

    // Check if GPU directory already exists
    if (uvm_id_gpu_index(gpu_id) >= GVM_MAX_PROCESSORS || uvm_id_gpu_index(gpu_id) < 0)
        return -EINVAL;

    gpu_debugfs = &proc_debugfs->gpus[uvm_id_gpu_index(gpu_id)];
    if (gpu_debugfs->gpu_dir)
        return 0;  // Already exists

    if (va_space) {
        UVM_ASSERT(va_space->gpu_cgroup != NULL);
        va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].memory_limit = -1ULL;
        va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].memory_current = 0;
        va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].memory_swap_current = 0;

        va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].compute_priority = GVM_MIN_PRIORITY / 2;
        va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].compute_freeze = 0;
    }

    // Initialize GPU debugfs entry
    gpu_debugfs->pid = pid;
    gpu_debugfs->gpu_id = gpu_id;

    // Create GPU subdirectory
    snprintf(gpu_dirname, sizeof(gpu_dirname), "%d", uvm_id_gpu_index(gpu_id));
    gpu_debugfs->gpu_dir = debugfs_create_dir(gpu_dirname, proc_debugfs->process_dir);
    if (!gpu_debugfs->gpu_dir) {
        ret = -ENOMEM;
        goto cleanup;
    }

    // Create files in GPU directory
    gpu_debugfs->memory_limit = debugfs_create_file("memory.limit", 0644, gpu_debugfs->gpu_dir,
                                                   gpu_debugfs, &gvm_process_memory_limit_fops);
    if (!gpu_debugfs->memory_limit) {
        ret = -ENOMEM;
        goto cleanup;
    }

    gpu_debugfs->memory_current =
        debugfs_create_file("memory.current", 0444, gpu_debugfs->gpu_dir, gpu_debugfs,
                            &gvm_process_memory_current_fops);
    if (!gpu_debugfs->memory_current) {
        ret = -ENOMEM;
        goto cleanup;
    }

    gpu_debugfs->memory_swap_current =
        debugfs_create_file("memory.swap.current", 0444, gpu_debugfs->gpu_dir, gpu_debugfs,
                            &gvm_process_memory_swap_current_fops);
    if (!gpu_debugfs->memory_swap_current) {
        ret = -ENOMEM;
        goto cleanup;
    }

    gpu_debugfs->compute_priority = debugfs_create_file("compute.priority", 0644, gpu_debugfs->gpu_dir,
                                                    gpu_debugfs, &gvm_process_compute_priority_fops);
    if (!gpu_debugfs->compute_priority) {
        ret = -ENOMEM;
        goto cleanup;
    }

    gpu_debugfs->compute_freeze = debugfs_create_file("compute.freeze", 0644, gpu_debugfs->gpu_dir,
                                                    gpu_debugfs, &gvm_process_compute_freeze_fops);
    if (!gpu_debugfs->compute_freeze) {
        ret = -ENOMEM;
        goto cleanup;
    }

    gpu_debugfs->gcgroup_stat =
        debugfs_create_file("gcgroup.stat", 0444, gpu_debugfs->gpu_dir, gpu_debugfs,
                            &gvm_process_gcgroup_stat_fops);
    if (!gpu_debugfs->gcgroup_stat) {
        ret = -ENOMEM;
        goto cleanup;
    }

    proc_debugfs->num_gpus_created++;
    return 0;

cleanup:
    if (gpu_debugfs->gpu_dir) {
        debugfs_remove_recursive(gpu_debugfs->gpu_dir);
        gpu_debugfs->gpu_dir = NULL;
    }
    return ret;
}

int gvm_debugfs_remove_gpu_dir(pid_t pid, uvm_gpu_id_t gpu_id)
{
    struct gvm_process_debugfs *proc_debugfs = NULL;
    struct gvm_gpu_debugfs *gpu_debugfs;

    // Find the process debugfs entry
    spin_lock(&gvm_debugfs_lock);
    hash_for_each_possible(gvm_debugfs_dirs, proc_debugfs, hash_node, pid)
    {
        if (proc_debugfs->pid == pid) {
            break;
        }
        proc_debugfs = NULL;
    }
    spin_unlock(&gvm_debugfs_lock);

    if (!proc_debugfs)
        return -ENOENT;

    // Check if GPU directory already exists
    if (uvm_id_gpu_index(gpu_id) >= GVM_MAX_PROCESSORS || uvm_id_gpu_index(gpu_id) < 0)
        return -EINVAL;

    gpu_debugfs = &proc_debugfs->gpus[uvm_id_gpu_index(gpu_id)];
    if (!gpu_debugfs->gpu_dir)
        return -ENOENT;

    // Remove GPU directory
    debugfs_remove_recursive(gpu_debugfs->gpu_dir);
    gpu_debugfs->gpu_dir = NULL;

    proc_debugfs->num_gpus_created--;
    return 0;
}

//
// Main debugfs interface
//

int gvm_debugfs_init(void)
{
    // Create root directory
    gvm_debugfs_root = debugfs_create_dir("nvidia-uvm", NULL);
    if (!gvm_debugfs_root)
        return -ENOMEM;

    // Create processes directory
    gvm_debugfs_processes_dir = debugfs_create_dir("processes", gvm_debugfs_root);
    if (!gvm_debugfs_processes_dir)
        goto cleanup_root;

    // Create global process list file
    if (!debugfs_create_file("list", 0444, gvm_debugfs_processes_dir, NULL,
                             &gvm_processes_list_fops))
        goto cleanup_processes;

    return 0;

cleanup_processes:
    debugfs_remove_recursive(gvm_debugfs_processes_dir);
    gvm_debugfs_processes_dir = NULL;
cleanup_root:
    debugfs_remove_recursive(gvm_debugfs_root);
    gvm_debugfs_root = NULL;
    return -ENOMEM;
}

void gvm_debugfs_exit(void)
{
    struct gvm_process_debugfs *proc_debugfs;
    struct hlist_node *tmp;
    int bucket;

    // Remove all per-process directories
    spin_lock(&gvm_debugfs_lock);
    hash_for_each_safe(gvm_debugfs_dirs, bucket, tmp, proc_debugfs, hash_node)
    {
        hash_del(&proc_debugfs->hash_node);
        debugfs_remove_recursive(proc_debugfs->process_dir);
        kfree(proc_debugfs);
    }
    spin_unlock(&gvm_debugfs_lock);

    // Remove root directories
    debugfs_remove_recursive(gvm_debugfs_root);
    gvm_debugfs_root = NULL;
    gvm_debugfs_processes_dir = NULL;
}

//
// Util functions
//

static uvm_va_space_t *_gvm_find_va_space_by_pid(pid_t pid)
{
    uvm_va_space_t *va_space_out = NULL;
    uvm_va_space_t *va_space;

    uvm_mutex_lock(&g_uvm_global.va_spaces.lock);
    list_for_each_entry(va_space, &g_uvm_global.va_spaces.list, list_node) {
        if (va_space->pid == pid) {
            va_space_out = va_space;
            break;
        }
    }
    uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);

    return va_space_out;
}

static uvm_gpu_cgroup_t *_gvm_find_gpu_cgroup_by_pid(pid_t pid)
{
    uvm_gpu_cgroup_t *gpu_cgroup = NULL;
    uvm_va_space_t *va_space;

    uvm_mutex_lock(&g_uvm_global.va_spaces.lock);
    list_for_each_entry(va_space, &g_uvm_global.va_spaces.list, list_node) {
        if (va_space->pid == pid) {
            gpu_cgroup = va_space->gpu_cgroup;
            break;
        }
    }
    uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);

    return gpu_cgroup;
}

static size_t _gvm_find_and_acquire_va_spaces_by_pid(pid_t pid, uvm_va_space_t **va_spaces, size_t size)
{
    uvm_va_space_t *va_space;
    size_t count = 0;

    uvm_mutex_lock(&g_uvm_global.va_spaces.lock);
    list_for_each_entry(va_space, &g_uvm_global.va_spaces.list, list_node) {
        if (count >= size)
            break;

        if (va_space->pid == pid) {
            atomic64_add(1, &va_space->num_debugfs_refs);
            va_spaces[count] = va_space;
            count += 1;
        }
    }
    uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);

    return count;
}

static size_t _gvm_release_va_spaces(uvm_va_space_t **va_spaces, size_t size) {
    size_t index;

    for (index = 0; index < size; ++index) {
        atomic64_sub(1, &va_spaces[index]->num_debugfs_refs);
    }

    return size;
}

int try_charge_gpu_memcg_debugfs(uvm_va_space_t *va_space, uvm_gpu_id_t gpu_id, size_t size, bool swap) {
    UVM_ASSERT(va_space->gpu_cgroup);
    size_t *memcg_current = (swap) ? &(va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].memory_swap_current) :
        &(va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].memory_current);
    *memcg_current += size;
    return 0;
}

int try_uncharge_gpu_memcg_debugfs(uvm_va_space_t *va_space, uvm_gpu_id_t gpu_id, size_t size, bool swap) {
    UVM_ASSERT(va_space->gpu_cgroup);
    size_t *memcg_current = (swap) ? &(va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].memory_swap_current) :
        &(va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].memory_current);
    if (*memcg_current > size) {
        *memcg_current -= size;
    }
    else {
        *memcg_current = 0;
    }
    return 0;
}

size_t get_gpu_memcg_current(uvm_va_space_t *va_space, uvm_gpu_id_t gpu_id) {
    UVM_ASSERT(va_space->gpu_cgroup);
    return va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].memory_current;
}

size_t get_gpu_memcg_limit(uvm_va_space_t *va_space, uvm_gpu_id_t gpu_id) {
    UVM_ASSERT(va_space->gpu_cgroup);
    return va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].memory_limit;
}

NV_STATUS gvm_update_event_count(UVM_UPDATE_EVENT_COUNT_PARAMS *params, uvm_va_space_t *va_space, uvm_gpu_id_t gpu_id) {
    UVM_ASSERT(va_space->gpu_cgroup);

    if (params->type == UVM_SUBMIT_KERNEL_EVENT) {
        if (params->op == UVM_ADD_EVENT_COUNT) {
            atomic64_add(params->value, &va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].nr_submitted_kernels);
        } else if (params->op == UVM_SET_EVENT_COUNT) {
            atomic64_set(&va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].nr_submitted_kernels, params->value);
        } else {
            return NV_ERR_INVALID_ARGUMENT;
        }
    } else if (params->type == UVM_END_KERNEL_EVENT) {
        if (params->op == UVM_ADD_EVENT_COUNT) {
            atomic64_add(params->value, &va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].nr_ended_kernels);
        } else if (params->op == UVM_SET_EVENT_COUNT) {
            atomic64_set(&va_space->gpu_cgroup[uvm_id_gpu_index(gpu_id)].nr_ended_kernels, params->value);
        } else {
            return NV_ERR_INVALID_ARGUMENT;
        }
    } else {
        return NV_ERR_INVALID_ARGUMENT;
    }

    return NV_OK;
}
