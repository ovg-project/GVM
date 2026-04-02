package gvm

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// FindGPUPIDs returns the list of PIDs that are GPU processes belonging to the
// given container PID (i.e., the container's init process or any of its descendants).
func FindGPUPIDs(containerPID int) ([]int, error) {
	// Build the full PID tree for this container
	allPIDs := []int{containerPID}
	children := getChildProcesses(containerPID)
	allPIDs = append(allPIDs, children...)

	// Read the nvidia-uvm processes directory
	entries, err := os.ReadDir(GVMProcessesPath)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", GVMProcessesPath, err)
	}

	// Build a set for O(1) lookup
	pidSet := make(map[int]bool, len(allPIDs))
	for _, p := range allPIDs {
		pidSet[p] = true
	}

	// Match GPU PIDs against container PID tree
	var gpuPIDs []int
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		pid, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}
		if pidSet[pid] {
			gpuPIDs = append(gpuPIDs, pid)
		}
	}

	return gpuPIDs, nil
}

// ListAllGPUPIDs returns all PIDs that currently have GPU processes.
func ListAllGPUPIDs() ([]int, error) {
	entries, err := os.ReadDir(GVMProcessesPath)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", GVMProcessesPath, err)
	}

	var pids []int
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		pid, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}
		pids = append(pids, pid)
	}
	return pids, nil
}

// getChildProcesses recursively finds all descendant PIDs of the given PID
// by scanning /proc/*/stat for parent PID matches.
func getChildProcesses(pid int) []int {
	var children []int

	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		childPID, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}

		statPath := filepath.Join("/proc", entry.Name(), "stat")
		data, err := os.ReadFile(statPath)
		if err != nil {
			continue
		}

		// Parse stat file: pid (comm) state ppid ...
		// The comm field can contain spaces and parentheses, so find the last ')' first.
		content := string(data)
		lastParen := strings.LastIndex(content, ")")
		if lastParen == -1 || lastParen+2 >= len(content) {
			continue
		}
		fieldsAfterComm := strings.Fields(content[lastParen+2:])
		if len(fieldsAfterComm) < 2 {
			continue
		}
		// fieldsAfterComm[0] = state, fieldsAfterComm[1] = ppid
		ppid, err := strconv.Atoi(fieldsAfterComm[1])
		if err != nil {
			continue
		}

		if ppid == pid {
			children = append(children, childPID)
			grandchildren := getChildProcesses(childPID)
			children = append(children, grandchildren...)
		}
	}

	return children
}
