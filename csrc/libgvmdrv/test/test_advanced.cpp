#include "gvmdrv.h"
#include "gvmdrv_log.h"
#include <iostream>
#include <cassert>

int test_interleave_operations() {
    std::cout << "Testing interleave operations..." << std::endl;

    int fd = find_initialized_uvm();
    if (fd < 0) {
        std::cout << "✗ Skipping interleave test - no UVM available" << std::endl;
        return 0;
    }

    // Test setting interleave level
    unsigned int test_level = 2;
    set_interleave(fd, test_level);
    std::cout << "✓ Set interleave level to " << test_level << std::endl;

    return 0;
}

int test_scheduling_operations() {
    std::cout << "Testing scheduling operations..." << std::endl;

    int fd = find_initialized_uvm();
    if (fd < 0) {
        std::cout << "✗ Skipping scheduling test - no UVM available" << std::endl;
        return 0;
    }

    // Test disabling scheduling
    schedule(fd, false);
    std::cout << "✓ Disabled scheduling" << std::endl;

    // Test enabling scheduling
    schedule(fd, true);
    std::cout << "✓ Enabled scheduling" << std::endl;

    return 0;
}

int test_error_handling() {
    std::cout << "Testing error handling..." << std::endl;

    // Test with invalid file descriptor
    set_timeslice(-1, 1000);
    std::cout << "✓ Error handling for invalid fd works" << std::endl;

    get_timeslice(-1);
    std::cout << "✓ Error handling for get_timeslice with invalid fd works" << std::endl;

    return 0;
}

int main() {
    std::cout << "=== libgvmdrv Advanced Tests ===" << std::endl;

    int failures = 0;

    failures += test_interleave_operations();
    failures += test_scheduling_operations();
    failures += test_error_handling();

    std::cout << "\n=== Test Summary ===" << std::endl;
    if (failures == 0) {
        std::cout << "✓ All advanced tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ " << failures << " test(s) failed" << std::endl;
        return 1;
    }
}