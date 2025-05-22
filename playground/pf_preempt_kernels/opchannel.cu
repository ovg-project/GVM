#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>

#define NVA06C_CTRL_CMD_SET_TIMESLICE (0xa06c0103) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GROUP_A_GPFIFO_INTERFACE_ID << 8) | NVA06C_CTRL_SET_TIMESLICE_PARAMS_MESSAGE_ID" */
#define NVA06C_CTRL_CMD_GET_TIMESLICE (0xa06c0104) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GROUP_A_GPFIFO_INTERFACE_ID << 8) | NVA06C_CTRL_GET_TIMESLICE_PARAMS_MESSAGE_ID" */
#define NVA06F_CTRL_CMD_STOP_CHANNEL (0xa06f0112) /* finn: Evaluated from "(FINN_KEPLER_CHANNEL_GPFIFO_A_GPFIFO_INTERFACE_ID << 8) | NVA06F_CTRL_STOP_CHANNEL_PARAMS_MESSAGE_ID" */

#define UVM_OPERATE_USER_CHANNELS 81
typedef struct
{
	unsigned int			cmd;       // IN
	union {
		long long unsigned timesliseUs;
		bool immediate;
	} data;							   // IN
	size_t 					dataSize;  // IN
	int						rmStatus;  // OUT
} UVM_OPERATE_USER_CHANNELS_PARAMS;

#define UVM_IS_INITIALIZED 82
typedef struct
{
	bool					initialized;    // IN/OUT
	int						rmStatus;		// OUT
} UVM_IS_INITIALIZED_PARAMS;

__global__ void quick_kernel() {
	float acc = threadIdx.x;
    #pragma unroll 100
    for (int i = 0; i < 10000000; ++i) {
        acc = sinf(acc) * cosf(acc) + acc;
    }
}

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

void get_timeslice(int fd) {
	if (fd < 0) {
		std::cerr << "Invalid file descriptor\n";
		return;
	}

	UVM_OPERATE_USER_CHANNELS_PARAMS params = {
		.cmd = NVA06F_CTRL_CMD_STOP_CHANNEL,
		.data = {
			.immediate = 0
		},
		.dataSize = sizeof(bool),
		.rmStatus = 0
	};
	int ret = ioctl(fd, UVM_OPERATE_USER_CHANNELS, &params);
	if (ret < 0) {
		std::cerr << "Failed to get timeslice: " << strerror(errno) << "\n";
	} else {
		std::cout << "Timeslice: " << params.data.timesliseUs << "\n";
	}
}

int main() {
	std::cout << "Press enter to start kernel execution...\n";
	std::cin.get();
	quick_kernel<<<1, 1>>>();

	int fd = find_initialized_uvm();
	if (fd < 0) {
		std::cerr << "Failed to find initialized UVM\n";
		return 1;
	}
	std::cout << "Found initialized UVM with fd: " << fd << "\n";
	std::cout << "Got timeslice...\n";
	get_timeslice(fd);

    cudaDeviceSynchronize();

	std::cin.get();

    return 0;
}
