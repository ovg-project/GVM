#include <linux/debugfs.h>
#include <linux/uaccess.h>
#include <linux/seq_file.h>
#include <linux/pid.h>
#include <linux/sched/signal.h>
#include <linux/hashtable.h>
#include <linux/slab.h>

#include "uvm_common.h"
#include "gvm_debugfs.h"
#include "uvm_global.h"
#include "uvm_linux.h"

// Global debugfs root directory
static struct dentry *gvm_debugfs_root;
static struct dentry *gvm_debugfs_processes_dir;

// Hash table for per-process debugfs directories
#define GVM_DEBUGFS_HASH_BITS 8
static DEFINE_HASHTABLE(gvm_debugfs_dirs, GVM_DEBUGFS_HASH_BITS);
static DEFINE_SPINLOCK(gvm_debugfs_lock);

//
// Declare util functions
//

static struct task_struct *gvm_find_task_by_pid(pid_t pid);
static int gvm_get_active_gpu_count(void);

//
// Per-process debugfs file operations
//

// Show memory limit for a specific process and GPU
static int gvm_process_memory_high_show(struct seq_file *m, void *data)
{
    struct gvm_gpu_debugfs *gpu_debugfs = m->private;

    // Should read from the metadata datastructure in target pid's uvm_va_space.
    // Return dummy value based on PID and GPU ID for demonstration
    size_t dummy_limit = (gpu_debugfs->pid * 1000) + (gpu_debugfs->gpu_id);

    seq_printf(m, "%zu\n", dummy_limit);
    return 0;
}

// Set memory limit for a specific process and GPU
static ssize_t gvm_process_memory_high_write(struct file *file, const char __user *user_buf,
                                             size_t count, loff_t *ppos)
{
    struct seq_file *m = file->private_data;
    struct gvm_gpu_debugfs *gpu_debugfs = m->private;
    char buf[32];
    size_t limit;
    int parsed;

    if (count >= sizeof(buf))
        return -EINVAL;

    if (copy_from_user(buf, user_buf, count))
        return -EFAULT;

    buf[count] = '\0';

    parsed = kstrtoul(buf, 10, (unsigned long *) &limit);
    if (parsed != 0)
        return -EINVAL;

    // gvm_set_gpu_memory_limit(gpu_debugfs->pid, gpu_debugfs->gpu_id, limit);
    pr_err("gvm_process_memory_high_write: pid=%d, gpu=%d, limit=%zu\n", gpu_debugfs->pid,
           gpu_debugfs->gpu_id, limit);
    return count;
}

// Show current memory usage for a specific process and GPU
static int gvm_process_memory_current_show(struct seq_file *m, void *data)
{
    struct gvm_gpu_debugfs *gpu_debugfs = m->private;

    // Should read from the metadata datastructure in target pid's uvm_va_space.
    // Return dummy value based on PID and GPU ID for demonstration
    size_t dummy_current = (gpu_debugfs->pid * 500) + (gpu_debugfs->gpu_id * 100);

    seq_printf(m, "%zu\n", dummy_current);
    return 0;
}

// Preempt a specific process on a specific GPU
static ssize_t gvm_process_compute_high_write(struct file *file, const char __user *user_buf,
                                              size_t count, loff_t *ppos)
{
    struct seq_file *m = file->private_data;
    struct gvm_gpu_debugfs *gpu_debugfs = m->private;
    struct task_struct *task;
    struct pid *pid_struct;

    // Find the task by PID
    rcu_read_lock();
    pid_struct = find_pid_ns(gpu_debugfs->pid, &init_pid_ns);
    if (pid_struct) {
        task = pid_task(pid_struct, PIDTYPE_PID);
        if (task)
            get_task_struct(task);
    } else {
        task = NULL;
    }
    rcu_read_unlock();

    if (!task)
        return -ESRCH;

    pr_err("gvm_process_compute_high_write: pid=%d, gpu=%d\n", gpu_debugfs->pid,
           gpu_debugfs->gpu_id);
    // // Get UVM file descriptors for this task
    // num_uvmfds = gvm_linux_api_get_task_uvmfd(task, uvmfds, 8);
    // if (num_uvmfds > 0) {
    //     // Preempt all UVM contexts
    //     for (i = 0; i < num_uvmfds; i++) {
    //         gvm_linux_api_preempt_task(task, uvmfds[i]);
    //     }
    // }

    put_task_struct(task);
    return count;
}

// Reschedule a specific process on a specific GPU
static ssize_t gvm_process_compute_current_write(struct file *file, const char __user *user_buf,
                                                 size_t count, loff_t *ppos)
{
    struct seq_file *m = file->private_data;
    struct gvm_gpu_debugfs *gpu_debugfs = m->private;
    struct task_struct *task;
    struct pid *pid_struct;

    // Find the task by PID
    rcu_read_lock();
    pid_struct = find_pid_ns(gpu_debugfs->pid, &init_pid_ns);
    if (pid_struct) {
        task = pid_task(pid_struct, PIDTYPE_PID);
        if (task)
            get_task_struct(task);
    } else {
        task = NULL;
    }
    rcu_read_unlock();

    if (!task)
        return -ESRCH;

    pr_err("gvm_process_compute_current_write: pid=%d, gpu=%d\n", gpu_debugfs->pid,
           gpu_debugfs->gpu_id);

    // Get UVM file descriptors for this task
    // num_uvmfds = gvm_linux_api_get_task_uvmfd(task, uvmfds, 8);
    // if (num_uvmfds > 0) {
    //     // Reschedule all UVM contexts
    //     for (i = 0; i < num_uvmfds; i++) {
    //         gvm_linux_api_reschedule_task(task, uvmfds[i]);
    //     }
    // }

    put_task_struct(task);
    return count;
}

//
// File operation structures
//

static int gvm_process_memory_high_open(struct inode *inode, struct file *file)
{
    return single_open(file, gvm_process_memory_high_show, inode->i_private);
}

static const struct file_operations gvm_process_memory_high_fops = {
    .open = gvm_process_memory_high_open,
    .read = seq_read,
    .write = gvm_process_memory_high_write,
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

static int gvm_process_compute_high_open(struct inode *inode, struct file *file)
{
    return single_open(file, NULL, inode->i_private);
}

static const struct file_operations gvm_process_compute_high_fops = {
    .open = gvm_process_compute_high_open,
    .write = gvm_process_compute_high_write,
    .llseek = seq_lseek,
    .release = single_release,
};

static int gvm_process_compute_current_open(struct inode *inode, struct file *file)
{
    return single_open(file, NULL, inode->i_private);
}

static const struct file_operations gvm_process_compute_current_fops = {
    .open = gvm_process_compute_current_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = single_release,
};

//
// Global process list (for debugging)
//

static int gvm_processes_list_show(struct seq_file *m, void *data)
{
    seq_printf(m, "PID\tGPU_ID\tMemory_Limit\tMemory_Current\tLast_Update\n");
    seq_printf(m, "1234\t0\t1000000\t500000\t12345\n");
    seq_printf(m, "1234\t1\t2000000\t750000\t12346\n");
    seq_printf(m, "5678\t0\t1500000\t600000\t12347\n");
    seq_printf(m, "# This is a dummy list showing per-GPU process entries\n");

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

static int gvm_get_active_gpu_count(void);  // Implemented at the end.

int gvm_debugfs_create_process_dir(pid_t pid)
{
    struct gvm_process_debugfs *proc_debugfs;
    struct gvm_gpu_debugfs *gpu_debugfs;
    char process_dirname[16];
    char gpu_dirname[16];
    int gpu_id;
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

    // Create GPU subdirectories and files
    for (gpu_id = 0; gpu_id < gvm_get_active_gpu_count(); gpu_id++) {
        gpu_debugfs = &proc_debugfs->gpus[gpu_id];
        gpu_debugfs->pid = pid;
        gpu_debugfs->gpu_id = gpu_id;

        // Create GPU subdirectory
        snprintf(gpu_dirname, sizeof(gpu_dirname), "%d", gpu_id);
        gpu_debugfs->gpu_dir = debugfs_create_dir(gpu_dirname, proc_debugfs->process_dir);
        if (!gpu_debugfs->gpu_dir) {
            ret = -ENOMEM;
            goto cleanup_dirs;
        }

        // Create files in GPU directory
        gpu_debugfs->memory_high = debugfs_create_file("memory.high", 0644, gpu_debugfs->gpu_dir,
                                                       gpu_debugfs, &gvm_process_memory_high_fops);
        if (!gpu_debugfs->memory_high) {
            ret = -ENOMEM;
            goto cleanup_dirs;
        }

        gpu_debugfs->memory_current =
            debugfs_create_file("memory.current", 0444, gpu_debugfs->gpu_dir, gpu_debugfs,
                                &gvm_process_memory_current_fops);
        if (!gpu_debugfs->memory_current) {
            ret = -ENOMEM;
            goto cleanup_dirs;
        }

        gpu_debugfs->compute_high =
            debugfs_create_file("compute.high", 0644, gpu_debugfs->gpu_dir, gpu_debugfs,
                                &gvm_process_compute_high_fops);
        if (!gpu_debugfs->compute_high) {
            ret = -ENOMEM;
            goto cleanup_dirs;
        }

        gpu_debugfs->compute_current =
            debugfs_create_file("compute.current", 0444, gpu_debugfs->gpu_dir, gpu_debugfs,
                                &gvm_process_compute_current_fops);
        if (!gpu_debugfs->compute_current) {
            ret = -ENOMEM;
            goto cleanup_dirs;
        }

        proc_debugfs->num_gpus_created++;
    }

    // Add to hash table
    spin_lock(&gvm_debugfs_lock);
    hash_add(gvm_debugfs_dirs, &proc_debugfs->hash_node, pid);
    spin_unlock(&gvm_debugfs_lock);

    return 0;

cleanup_dirs:
    debugfs_remove_recursive(proc_debugfs->process_dir);
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

// Find the task by PID
static struct task_struct *gvm_find_task_by_pid(pid_t pid)
{
    struct task_struct *task = NULL;
    struct pid *pid_struct;

    rcu_read_lock();
    pid_struct = find_pid_ns(pid, &init_pid_ns);
    if (pid_struct) {
        task = pid_task(pid_struct, PIDTYPE_PID);
        if (task)
            get_task_struct(task);
    }
    rcu_read_unlock();

    return task;
}

// Get count of active GPUs known to UVM
static int gvm_get_active_gpu_count(void)
{
    int count = 0;
    uvm_mutex_lock(&g_uvm_global.global_lock);
    count = uvm_processor_mask_get_gpu_count(&g_uvm_global.retained_gpus);
    uvm_mutex_unlock(&g_uvm_global.global_lock);
    return count;
}