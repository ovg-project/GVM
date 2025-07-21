/**
 * @file libgvmdrv.h
 * @brief A minimalistic interface set for interacting with the NVIDIA GPU
 * Virtual Memory (UVM) driver with GVM extended functionality. Working with
 * both C and C++ code.
 */

#ifndef __LIBGVMDRV_H__
#define __LIBGVMDRV_H__

#ifdef __cplusplus
extern "C" {
#endif

int gvm_find_initialized_uvm();
void gvm_set_timeslice(int fd, long long unsigned timesliceUs);
long long unsigned gvm_get_timeslice(int fd);
void gvm_preempt(int fd);
void gvm_restart(int fd);
void gvm_schedule(int fd, bool enable);
void gvm_stop(int fd);
void gvm_set_interleave(int fd, unsigned int interleave);
void gvm_bind(int fd);
void gvm_set_gmemcg(int fd, unsigned long long size);

#ifdef __cplusplus
}
#endif

#endif // __LIBGVMDRV_H__
