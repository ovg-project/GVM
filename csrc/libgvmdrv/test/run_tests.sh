#!/bin/bash

# Test runner script for libgvmdrv
# Usage: ./run_tests.sh [test_name]

set -e

SCRIPT_DIR=$(dirname $(readlink -f "$0"))

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test directory
TEST_DIR="$SCRIPT_DIR"
BUILD_DIR="$SCRIPT_DIR/../build"
TEST_BUILD_DIR="$BUILD_DIR/test"

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "PASS")
            echo -e "${GREEN}✓ PASS${NC}: $message"
            ;;
        "FAIL")
            echo -e "${RED}✗ FAIL${NC}: $message"
            ;;
        "SKIP")
            echo -e "${YELLOW}⚠ SKIP${NC}: $message"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ INFO${NC}: $message"
            ;;
    esac
}

# Function to run a single test
run_test() {
    local test_name=$1
    local test_path="$TEST_BUILD_DIR/$test_name"

    if [[ ! -f "$test_path" ]]; then
        print_status "FAIL" "Test executable not found: $test_path"
        return 1
    fi

    if [[ ! -x "$test_path" ]]; then
        print_status "FAIL" "Test executable not executable: $test_path"
        return 1
    fi

    print_status "INFO" "Running $test_name..."

    # Run the test and capture output and exit code
    local output
    local exit_code
    output=$("$test_path" 2>&1)
    exit_code=$?

    # Print test output
    echo "$output"

    if [[ $exit_code -eq 0 ]]; then
        print_status "PASS" "$test_name completed successfully"
        return 0
    else
        print_status "FAIL" "$test_name failed with exit code $exit_code"
        return 1
    fi
}

# Main execution
main() {
    echo -e "${BLUE}=== libgvmdrv Test Suite ===${NC}"
    echo

    # Check if build directory exists
    if [[ ! -d "$BUILD_DIR" ]]; then
        print_status "FAIL" "Build directory not found. Run 'make' first."
        exit 1
    fi

    # Check if test build directory exists
    if [[ ! -d "$TEST_BUILD_DIR" ]]; then
        print_status "FAIL" "Test build directory not found. Run 'make test' first."
        exit 1
    fi

    pushd $BUILD_DIR  # switch to build directory to load the library

    local total_tests=0
    local passed_tests=0
    local failed_tests=0

    # If a specific test is requested
    if [[ $# -eq 1 ]]; then
        local test_name=$1
        if run_test "$test_name"; then
            passed_tests=1
        else
            failed_tests=1
        fi
        total_tests=1
    else
        # Run all tests
        local tests=("test" "test_basic" "test_advanced")

        for test_name in "${tests[@]}"; do
            total_tests=$((total_tests + 1))
            if run_test "$test_name"; then
                passed_tests=$((passed_tests + 1))
            else
                failed_tests=$((failed_tests + 1))
            fi
            echo
        done
    fi

    popd  # switch back to the original directory

    # Print summary
    echo -e "${BLUE}=== Test Summary ===${NC}"
    echo "Total tests: $total_tests"
    echo -e "Passed: ${GREEN}$passed_tests${NC}"
    echo -e "Failed: ${RED}$failed_tests${NC}"

    if [[ $failed_tests -eq 0 ]]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed!${NC}"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"