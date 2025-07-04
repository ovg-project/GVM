#include "gvmdrv.h"
#include "gvmdrv_log.h"
#include <iostream>
#include <cassert>

int test_find_initialized_uvm() {
    std::cout << "Testing find_initialized_uvm()..." << std::endl;

    int fd = find_initialized_uvm();
    if (fd >= 0) {
        std::cout << "✓ Found initialized UVM with fd: " << fd << std::endl;
        return 0;
    } else {
        std::cout << "✗ No initialized UVM found (this may be expected)" << std::endl;
        return 1; // Not necessarily a failure
    }
}

int test_timeslice_operations() {
    std::cout << "Testing timeslice operations..." << std::endl;

    int fd = find_initialized_uvm();
    if (fd < 0) {
        std::cout << "✗ Skipping timeslice test - no UVM available" << std::endl;
        return 0;
    }

    // Test getting current timeslice
    long long unsigned current = get_timeslice(fd);
    std::cout << "✓ Current timeslice: " << current << " us" << std::endl;

    // Test setting a new timeslice
    long long unsigned new_timeslice = current * 2;
    set_timeslice(fd, new_timeslice);

    // Verify the change
    long long unsigned verify = get_timeslice(fd);
    if (verify == new_timeslice) {
        std::cout << "✓ Timeslice successfully set to: " << verify << " us" << std::endl;
    } else {
        std::cout << "✗ Timeslice verification failed: expected " << new_timeslice
                  << ", got " << verify << std::endl;
        return 1;
    }

    // Restore original timeslice
    set_timeslice(fd, current);
    std::cout << "✓ Restored original timeslice" << std::endl;

    return 0;
}

int main() {
    std::cout << "=== libgvmdrv Basic Tests ===" << std::endl;

    int failures = 0;

    failures += test_find_initialized_uvm();
    failures += test_timeslice_operations();

    std::cout << "\n=== Test Summary ===" << std::endl;
    if (failures == 0) {
        std::cout << "✓ All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ " << failures << " test(s) failed" << std::endl;
        return 1;
    }
}
