#include "gvmdrv.h"
#include "gvmdrv_log.h"
#include <iostream>
#include <cstdlib>

int main() {
    std::cout << "Testing libgvmdrv library..." << std::endl;

    // Test finding initialized UVM
    int fd = find_initialized_uvm();
    if (fd >= 0) {
        std::cout << "Found initialized UVM with fd: " << fd << std::endl;

        // Test getting current timeslice
        long long unsigned current_timeslice = get_timeslice(fd);
        std::cout << "Current timeslice: " << current_timeslice << " us" << std::endl;

        // Test setting a new timeslice (double the current one)
        long long unsigned new_timeslice = current_timeslice * 2;
        std::cout << "Setting timeslice to: " << new_timeslice << " us" << std::endl;
        set_timeslice(fd, new_timeslice);

        // Verify the change
        long long unsigned verify_timeslice = get_timeslice(fd);
        std::cout << "Verified timeslice: " << verify_timeslice << " us" << std::endl;

        // Test setting interleave level
        std::cout << "Setting interleave level to 2" << std::endl;
        set_interleave(fd, 2);

        std::cout << "All tests completed successfully!" << std::endl;
        return 0;
    } else {
        std::cerr << "Failed to find initialized UVM. Make sure NVIDIA drivers are loaded and UVM is initialized." << std::endl;
        std::cerr << "You may need to run: sudo nvidia-modprobe -u -c=0" << std::endl;
        return 1;
    }
}