#include <cuda_runtime.h>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <sys/mman.h>

#define KERNEL_NOTIFICATION_SHM "/kernel_notification"
#define KERNEL_NOTIFICATION_SHM_SIZE 4096
struct kernel_notification {
	std::atomic<bool> running;
};

#define NVA06C_CTRL_CMD_SET_TIMESLICE (0xa06c0103) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GROUP_A_GPFIFO_INTERFACE_ID << 8) | NVA06C_CTRL_SET_TIMESLICE_PARAMS_MESSAGE_ID" */
#define NVA06C_CTRL_CMD_GET_TIMESLICE (0xa06c0104) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GROUP_A_GPFIFO_INTERFACE_ID << 8) | NVA06C_CTRL_GET_TIMESLICE_PARAMS_MESSAGE_ID" */
#define NVA06F_CTRL_CMD_STOP_CHANNEL (0xa06f0112) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) | NVA06F_CTRL_STOP_CHANNEL_PARAMS_MESSAGE_ID" */
#define NVA06C_CTRL_CMD_PREEMPT (0xa06c0105) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GROUP_A_GPFIFO_INTERFACE_ID << 8) | NVA06C_CTRL_PREEMPT_PARAMS_MESSAGE_ID" */
#define NVA06F_CTRL_CMD_GPFIFO_SCHEDULE (0xa06f0103) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) | NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS_MESSAGE_ID" */
#define NVA06F_CTRL_CMD_RESTART_RUNLIST (0xa06f0111) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) | NVA06F_CTRL_RESTART_RUNLIST_PARAMS_MESSAGE_ID" */
#define NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL (0xa06c0107) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GROUP_A_GPFIFO_INTERFACE_ID << 8) | NVA06C_CTRL_SET_INTERLEAVE_LEVEL_PARAMS_MESSAGE_ID" */
#define NVA06F_CTRL_CMD_BIND (0xa06f0104) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) | NVA06F_CTRL_BIND_PARAMS_MESSAGE_ID" */

#define UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP 81
#define UVM_CTRL_CMD_OPERATE_CHANNEL 82

typedef struct {
	std::uint64_t       timesliceUs;
} NVA06C_CTRL_TIMESLICE_PARAMS;

typedef struct {
    bool bWait;
    bool bManualTimeout;
	std::uint32_t  timeoutUs;
} NVA06C_CTRL_PREEMPT_PARAMS;

typedef struct {
    std::uint32_t tsgInterleaveLevel;     // IN
} NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS;

typedef union {
	NVA06C_CTRL_TIMESLICE_PARAMS NVA06C_CTRL_TIMESLICE_PARAMS;
	NVA06C_CTRL_PREEMPT_PARAMS NVA06C_CTRL_PREEMPT_PARAMS;
	NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS;
} UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS_DATA;

typedef struct {
    bool bEnable;               // IN
    bool bSkipSubmit;           // IN
    bool bSkipEnable;           // IN
} NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS;

typedef struct {
    bool bForceRestart;
    bool bBypassWait;
} NVA06F_CTRL_RESTART_RUNLIST_PARAMS;

typedef struct {
    bool bImmediate;
} NVA06F_CTRL_STOP_CHANNEL_PARAMS;

typedef struct NVA06F_CTRL_BIND_PARAMS {
	std::uint32_t engineType;
} NVA06F_CTRL_BIND_PARAMS;

typedef union {
	NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS;
	NVA06F_CTRL_RESTART_RUNLIST_PARAMS NVA06F_CTRL_RESTART_RUNLIST_PARAMS;
	NVA06F_CTRL_STOP_CHANNEL_PARAMS NVA06F_CTRL_STOP_CHANNEL_PARAMS;
	NVA06F_CTRL_BIND_PARAMS NVA06F_CTRL_BIND_PARAMS;
} UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS_DATA;

typedef struct
{
	unsigned int			cmd;       // IN
	UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS_DATA data;							   // IN
	size_t 					dataSize;  // IN
	int						rmStatus;  // OUT
} UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS;

typedef struct
{
	unsigned int			cmd;       // IN
	UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS_DATA data;							   // IN
	size_t 					dataSize;  // IN
	int						rmStatus;  // OUT
} UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS;

#define UVM_IS_INITIALIZED 80
typedef struct
{
	bool					initialized;    // IN/OUT
	int						rmStatus;		// OUT
} UVM_IS_INITIALIZED_PARAMS;

int find_initialized_uvm() {
	pid_t pid = getpid();
	const std::string target_path = "/dev/nvidia-uvm";

    std::string fd_dir = "/proc/" + std::to_string(pid) + "/fd";
    for (const auto& entry : std::filesystem::directory_iterator(fd_dir)) {
        std::error_code ec;
        std::string fd_path = std::filesystem::read_symlink(entry, ec).string();
        if (ec) continue;

		if (fd_path == target_path) {
			int fd = std::stoi(entry.path().filename());
			if (fd >= 0) {
				UVM_IS_INITIALIZED_PARAMS params = {
					.initialized = false,
					.rmStatus = 0
				};
				if (ioctl(fd, UVM_IS_INITIALIZED, &params) == 0) {
					if (params.initialized) {
						return fd;
					}
				} else {
					std::cerr << "Failed to check UVM initialization: " << strerror(errno) << "\n";
				}
			} else {
				std::cerr << "Invalid file descriptor: " << fd << "\n";
			}
		}
    }
    return -1;
}

void set_timeslice(int fd, long long unsigned timesliceUs) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS params = {
		.cmd = NVA06C_CTRL_CMD_SET_TIMESLICE,
		.data = {
			.NVA06C_CTRL_TIMESLICE_PARAMS = {
				.timesliceUs = timesliceUs
			}
		},
		.dataSize = sizeof(NVA06C_CTRL_TIMESLICE_PARAMS),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP, &params);
	if (ret < 0) {
		std::cerr << "Failed to set timeslice: " << strerror(errno) << "\n";
	} else {
		std::cout << "Timeslice: " << params.data.NVA06C_CTRL_TIMESLICE_PARAMS.timesliceUs << "\n";
	}
}

long long unsigned get_timeslice(int fd) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return 0;
	}

	long long unsigned timesliceUs = 0;

	UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS params = {
		.cmd = NVA06C_CTRL_CMD_GET_TIMESLICE,
		.data = {
			.NVA06C_CTRL_TIMESLICE_PARAMS = {
				.timesliceUs = timesliceUs
			}
		},
		.dataSize = sizeof(NVA06C_CTRL_TIMESLICE_PARAMS),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP, &params);
	if (ret < 0) {
		std::cerr << "Failed to get timeslice: " << strerror(errno) << "\n";
	}

	return params.data.NVA06C_CTRL_TIMESLICE_PARAMS.timesliceUs;
}

void preempt(int fd) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS params = {
		.cmd = NVA06C_CTRL_CMD_PREEMPT,
		.data = {
			.NVA06C_CTRL_PREEMPT_PARAMS = {
				.bWait = true,
				.bManualTimeout = false,
				.timeoutUs = 1000
			}
		},
		.dataSize = sizeof(NVA06C_CTRL_PREEMPT_PARAMS),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP, &params);
	if (ret < 0) {
		std::cerr << "Failed to preempt: " << strerror(errno) << "\n";
	} else {
		std::cout << "Preempted successfully" << "\n";
	}
}

void restart(int fd) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS params = {
		.cmd = NVA06F_CTRL_CMD_RESTART_RUNLIST,
		.data = {
			.NVA06F_CTRL_RESTART_RUNLIST_PARAMS = {
				.bForceRestart = true,
				.bBypassWait = false
			}
		},
		.dataSize = sizeof(NVA06F_CTRL_CMD_RESTART_RUNLIST),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL, &params);
	if (ret < 0) {
		std::cerr << "Failed to restart: " << strerror(errno) << "\n";
	} else {
		std::cout << "Restarted successfully" << "\n";
	}
}

void schedule(int fd, bool enable) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS params = {
		.cmd = NVA06F_CTRL_CMD_GPFIFO_SCHEDULE,
		.data = {
			.NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS = {
				.bEnable = enable,
				.bSkipSubmit = !enable,
				.bSkipEnable =!enable 
			}
		},
		.dataSize = sizeof(	NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL, &params);
	if (ret < 0) {
		std::cerr << "Failed to schedule: " << strerror(errno) << "\n";
	} else {
		std::cout << "Scheduled successfully" << "\n";
	}
}

void stop(int fd) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS params = {
		.cmd = NVA06F_CTRL_CMD_STOP_CHANNEL,
		.data = {
			.NVA06F_CTRL_STOP_CHANNEL_PARAMS = {
				.bImmediate = true
			}
		},
		.dataSize = sizeof(NVA06F_CTRL_STOP_CHANNEL_PARAMS),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL, &params);
	if (ret < 0) {
		std::cerr << "Failed to stop: " << strerror(errno) << "\n";
	} else {
		std::cout << "Stopped successfully" << "\n";
	}
}

void set_interleave(int fd, std::uint32_t interleave) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP_PARAMS params = {
		.cmd = NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL,
		.data = {
			.NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS = {
				.tsgInterleaveLevel = interleave
			}
		},
		.dataSize = sizeof(NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL_GROUP, &params);
	if (ret < 0) {
		std::cerr << "Failed to set interleave level: " << strerror(errno) << "\n";
	} else {
		std::cout << "Set interleave level to " << interleave << " successfully" << "\n";
	}
}

void bind(int fd) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_CTRL_CMD_OPERATE_CHANNEL_PARAMS params = {
		.cmd = NVA06F_CTRL_CMD_BIND,
		.data = {
			.NVA06F_CTRL_BIND_PARAMS = {
				// Will be filled in kernel module using restored type of engine
				.engineType = 0
			}
		},
		.dataSize = sizeof(NVA06F_CTRL_BIND_PARAMS),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_CTRL_CMD_OPERATE_CHANNEL, &params);
	if (ret < 0) {
		std::cerr << "Failed to bind: " << strerror(errno) << "\n";
	} else {
		std::cout << "Binded successfully" << "\n";
	}
}

__global__ void persistent_kernel(volatile int* stop_flag, float* dummy, unsigned clock_rate) {
	float acc = threadIdx.x;
	unsigned long long prev_timepoint;
	unsigned long long curr_timepoint = clock64();
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printf("Clock rate is %x\n", clock_rate);
	}
    while (!(*stop_flag)) {
		prev_timepoint = curr_timepoint;
		curr_timepoint = clock64();
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			if ((curr_timepoint - prev_timepoint) / clock_rate > 1000) {
				printf("Last cycle spent %lld ms\n", ((curr_timepoint - prev_timepoint) / clock_rate));
			}
		}
        for (int i = 0; i < 100000; ++i) {
            acc = sinf(acc) * cosf(acc) + acc;
        }
        // Write to global memory to prevent optimization
        dummy[0] = acc;
    }
}

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cerr << "Usage: ./persistent <timesliceUs>" << std::endl;
		return 1;
	}

	int shmfd = shm_open(KERNEL_NOTIFICATION_SHM, O_CREAT | O_RDWR, 0666);
	if (shmfd < 0) {
		std::cerr << "Failed to open shm" << std::endl;
		return 1;
	}
	ftruncate(shmfd, KERNEL_NOTIFICATION_SHM_SIZE);

	struct kernel_notification *notification = (struct kernel_notification *) mmap(NULL, KERNEL_NOTIFICATION_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shmfd, 0);
	if (!notification) {
		std::cerr << "Failed to map shm" << std::endl;
		return 1;
	}

	long long unsigned timesliceUs = std::stoull(argv[1]);

	std::size_t size = 4096;  // One memory page (4KB)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
	int device_id;
	cudaGetDevice(&device_id);

    std::cout << "GPU Name: " << prop.name << "\n";
    std::cout << "Total SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Max Threads/Block: " << prop.maxThreadsPerBlock << "\n";

    if (!prop.cooperativeLaunch) {
        std::cerr << "Cooperative launch not supported\n";
        return 1;
    }

    volatile int* stop_flag;
    cudaMallocManaged((void**)&stop_flag, sizeof(int));
    *stop_flag = 0;

	int uvmfd = find_initialized_uvm();
	set_timeslice(uvmfd, timesliceUs);

	std::cout << "Timeslice set to " << get_timeslice(uvmfd) << std::endl;

    const int block_size = 512;
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        persistent_kernel,
        block_size,
        0
    );

    int total_blocks = prop.multiProcessorCount * max_blocks_per_sm;

    std::cout << "Max active blocks per SM: " << max_blocks_per_sm << "\n";
    std::cout << "Launching total blocks: " << total_blocks
              << " (" << max_blocks_per_sm << " per SM × " << prop.multiProcessorCount << " SMs)\n";
    std::cout << "Threads per block: " << block_size
              << " → Total threads: " << total_blocks * block_size << "\n";

	float* dummy;
	cudaMallocManaged(&dummy, size);
	
	std::cout << "Press to start the persistent kernel...\n";
	std::cin.get();
	std::cout << "Starting persistent kernel...\n";
	
    std::thread input_thread([&]() {
        std::cout << "Press enter to stop persistent kernel...\n";
		std::cin.get();
        *stop_flag = 1;
    });

	printf("Outside clock rate is %x\n", prop.clockRate);
	void* args[] = { (void*)&stop_flag, &dummy, &prop.clockRate };

	notification->running.store(false);

	cudaLaunchCooperativeKernel(
	    (void*)persistent_kernel,
	    total_blocks,
	    block_size,
	    args,
	    0,
	    0
	);

	std::thread t = std::thread([notification, stop_flag, uvmfd] () {
		bool stopping = false;
		while (!*stop_flag) {
			if (notification->running.load()) {
				if (!stopping) {
					stopping = true;
					std::cerr << "New kernel inited, preempting myself" << std::endl;
					// std::chrono::high_resolution_clock::time_point start_stop = std::chrono::high_resolution_clock::now();
					// stop(uvmfd);
					// std::chrono::high_resolution_clock::time_point end_stop = std::chrono::high_resolution_clock::now();
					// std::cerr << "Stopped in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_stop - start_stop).count() << " ms" << std::endl;
					// restart(uvmfd);
					// schedule(uvmfd, false);
					for (int i = 0; i < 1; ++i) {
						preempt(uvmfd);
						std::this_thread::sleep_for(std::chrono::milliseconds(10));
					}
				}
			} else {
				if (stopping) {
					stopping = false;
					std::cerr << "New kernel stopped, rescheduling myself" << std::endl;
					// std::chrono::high_resolution_clock::time_point start_restart = std::chrono::high_resolution_clock::now();
					// bind(uvmfd);
					// for (int i = 5; i != 0; --i) {
					// 	std::cerr << "Channel will be scheduled in " << i << " seconds" << std::endl;
					// 	std::this_thread::sleep_for(std::chrono::milliseconds(1000));
					// }
					// schedule(uvmfd, true);
					// std::chrono::high_resolution_clock::time_point end_restart = std::chrono::high_resolution_clock::now();
					// std::cerr << "Restarted in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_restart - start_restart).count() << " ms" << std::endl;
				}
			}
		}
		std::cerr << "Standby thread exit" << std::endl;
	});

	while (!*stop_flag) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
	t.join();
    cudaDeviceSynchronize();
    input_thread.join();
	cudaFree(dummy);
    cudaFree((void*)stop_flag);

	munmap(notification, KERNEL_NOTIFICATION_SHM_SIZE);
    close(shmfd);
	shm_unlink(KERNEL_NOTIFICATION_SHM);

    return 0;
}
