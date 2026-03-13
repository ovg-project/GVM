package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	gvmProcessesPath = "/sys/kernel/debug/nvidia-uvm/processes"
	pollInterval     = 5 * time.Second
)

type ContainerInfo struct {
	ID     string
	PID    int
	Name   string
	EnvVars map[string]string
}

func main() {
	fmt.Println("GVM Docker Daemon starting...")
	
	// Track which containers we've already processed
	processedContainers := make(map[string]bool)
	
	for {
		containers, err := getRunningContainers()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error getting containers: %v\n", err)
			time.Sleep(pollInterval)
			continue
		}
		
		for _, container := range containers {
			// Skip if already processed
			if processedContainers[container.ID] {
				continue
			}
			
			// Check if container has GVM env vars
			memLimit := container.EnvVars["GVM_MEMORY_LIMIT"]
			priority := container.EnvVars["GVM_COMPUTE_PRIORITY"]
			
			if memLimit == "" && priority == "" {
				continue
			}
			
			fmt.Printf("[%s] Found GVM-enabled container: %s (PID: %d)\n", 
				time.Now().Format("15:04:05"), container.Name, container.PID)
			
			// Find GPU processes for this container
			gpuPIDs := findGPUProcesses(container.PID)
			if len(gpuPIDs) == 0 {
				fmt.Printf("[%s] No GPU processes yet for %s, will retry\n", 
					time.Now().Format("15:04:05"), container.Name)
				continue
			}
			
			// Apply GVM controls
			for _, gpuPID := range gpuPIDs {
				if err := applyGVMControls(gpuPID, memLimit, priority); err != nil {
					fmt.Fprintf(os.Stderr, "[%s] Error applying controls to PID %d: %v\n", 
						time.Now().Format("15:04:05"), gpuPID, err)
				} else {
					fmt.Printf("[%s] Applied GVM controls to PID %d (container: %s)\n", 
						time.Now().Format("15:04:05"), gpuPID, container.Name)
					processedContainers[container.ID] = true
				}
			}
		}
		
		time.Sleep(pollInterval)
	}
}

func getRunningContainers() ([]ContainerInfo, error) {
	cmd := exec.Command("docker", "ps", "-q")
	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	
	containerIDs := strings.Split(strings.TrimSpace(string(output)), "\n")
	var containers []ContainerInfo
	
	for _, id := range containerIDs {
		if id == "" {
			continue
		}
		
		// Get container details
		cmd := exec.Command("docker", "inspect", id)
		output, err := cmd.Output()
		if err != nil {
			continue
		}
		
		var inspectData []struct {
			ID    string `json:"Id"`
			Name  string `json:"Name"`
			State struct {
				Pid int `json:"Pid"`
			} `json:"State"`
			Config struct {
				Env []string `json:"Env"`
			} `json:"Config"`
		}
		
		if err := json.Unmarshal(output, &inspectData); err != nil {
			continue
		}
		
		if len(inspectData) == 0 {
			continue
		}
		
		data := inspectData[0]
		
		// Parse environment variables
		envVars := make(map[string]string)
		for _, env := range data.Config.Env {
			parts := strings.SplitN(env, "=", 2)
			if len(parts) == 2 {
				envVars[parts[0]] = parts[1]
			}
		}
		
		containers = append(containers, ContainerInfo{
			ID:      data.ID,
			PID:     data.State.Pid,
			Name:    strings.TrimPrefix(data.Name, "/"),
			EnvVars: envVars,
		})
	}
	
	return containers, nil
}

func findGPUProcesses(containerPID int) []int {
	// Get all child processes of the container
	allPIDs := []int{containerPID}
	children := getChildProcesses(containerPID)
	allPIDs = append(allPIDs, children...)
	
	// Check which PIDs have GPU processes
	var gpuPIDs []int
	entries, err := ioutil.ReadDir(gvmProcessesPath)
	if err != nil {
		return nil
	}
	
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		
		pid, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}
		
		// Check if this PID belongs to our container
		for _, containerPID := range allPIDs {
			if pid == containerPID {
				gpuPIDs = append(gpuPIDs, pid)
				break
			}
		}
	}
	
	return gpuPIDs
}

func getChildProcesses(pid int) []int {
	var children []int
	
	entries, err := ioutil.ReadDir("/proc")
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
		data, err := ioutil.ReadFile(statPath)
		if err != nil {
			continue
		}
		
		fields := strings.Fields(string(data))
		if len(fields) < 4 {
			continue
		}
		
		ppid, err := strconv.Atoi(fields[3])
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

func applyGVMControls(pid int, memLimit, priority string) error {
	basePath := filepath.Join(gvmProcessesPath, strconv.Itoa(pid), "0")
	
	// Set memory limit
	if memLimit != "" {
		limitPath := filepath.Join(basePath, "memory.limit")
		if err := ioutil.WriteFile(limitPath, []byte(memLimit), 0644); err != nil {
			return fmt.Errorf("failed to set memory limit: %w", err)
		}
		fmt.Printf("[%s] Set memory limit to %s for PID %d\n", 
			time.Now().Format("15:04:05"), memLimit, pid)
	}
	
	// Set compute priority
	if priority != "" {
		priorityPath := filepath.Join(basePath, "compute.priority")
		if err := ioutil.WriteFile(priorityPath, []byte(priority), 0644); err != nil {
			return fmt.Errorf("failed to set priority: %w", err)
		}
		fmt.Printf("[%s] Set compute priority to %s for PID %d\n", 
			time.Now().Format("15:04:05"), priority, pid)
	}
	
	return nil
}
