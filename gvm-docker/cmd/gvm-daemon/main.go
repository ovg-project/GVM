package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/ovg-project/gvm-docker/pkg/gvm"
)

const (
	pollInterval = 5 * time.Second
)

// containerState tracks the state of a container we've already processed.
type containerState struct {
	appliedPIDs map[int]bool // GPU PIDs we've already applied controls to
}

type containerInfo struct {
	ID      string
	PID     int
	Name    string
	EnvVars map[string]string
}

func main() {
	fmt.Println("GVM Docker Daemon v2 starting...")
	fmt.Printf("  Poll interval: %s\n", pollInterval)
	fmt.Printf("  GVM sysfs path: %s\n", gvm.GVMProcessesPath)

	// Query total GPU memory for percentage-based limits
	gpuMemory, err := gvm.QueryGPUTotalMemory()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: could not query GPU memory: %v\n", err)
		fmt.Fprintf(os.Stderr, "  GVM_MEMORY_PERCENTAGE will not work.\n")
		gpuMemory = make(map[int]int64)
	} else {
		for idx, mem := range gpuMemory {
			fmt.Printf("  GPU %d: %s total memory\n", idx, gvm.FormatMemoryValue(mem))
		}
	}
	fmt.Println()

	// Track container states (supports detecting new GPU processes in already-known containers)
	containerStates := make(map[string]*containerState)

	for {
		containers, err := getRunningContainers()
		if err != nil {
			fmt.Fprintf(os.Stderr, "[%s] Error listing containers: %v\n", ts(), err)
			time.Sleep(pollInterval)
			continue
		}

		// Track which container IDs are still running for cleanup
		activeIDs := make(map[string]bool)

		for _, c := range containers {
			activeIDs[c.ID] = true

			// Parse GVM config from container env vars
			config, err := gvm.ConfigFromEnv(c.EnvVars)
			if err != nil {
				fmt.Fprintf(os.Stderr, "[%s] Error parsing config for %s: %v\n", ts(), c.Name, err)
				continue
			}

			// Skip if GVM is disabled or no controls set
			if !config.Enabled || !config.HasAnyControl() {
				continue
			}

			// Initialize state tracking for this container
			if _, ok := containerStates[c.ID]; !ok {
				containerStates[c.ID] = &containerState{
					appliedPIDs: make(map[int]bool),
				}
				fmt.Printf("[%s] Discovered GVM container: %s (PID: %d)\n", ts(), c.Name, c.PID)
				if config.Debug {
					fmt.Printf("[%s]   Memory limit: %s\n", ts(), describeMemoryConfig(config))
					fmt.Printf("[%s]   Compute priority: %d\n", ts(), config.ComputePriority)
					fmt.Printf("[%s]   Compute freeze: %v\n", ts(), config.ComputeFreeze)
				}
			}

			state := containerStates[c.ID]

			// Find GPU processes for this container
			gpuPIDs, err := gvm.FindGPUPIDs(c.PID)
			if err != nil {
				if config.Debug {
					fmt.Fprintf(os.Stderr, "[%s] Error finding GPU PIDs for %s: %v\n", ts(), c.Name, err)
				}
				continue
			}

			if len(gpuPIDs) == 0 {
				continue
			}

			// Apply controls to any new GPU PIDs
			for _, gpuPID := range gpuPIDs {
				if state.appliedPIDs[gpuPID] {
					continue
				}

				// Discover GPU indices for this PID
				gpuIndices, err := gvm.ListGPUIndices(gpuPID)
				if err != nil {
					if config.Debug {
						fmt.Fprintf(os.Stderr, "[%s] Error listing GPU indices for PID %d: %v\n", ts(), gpuPID, err)
					}
					continue
				}

				allOK := true
				for _, gpuIdx := range gpuIndices {
					totalGPUMem := gpuMemory[gpuIdx]

					if err := gvm.ApplyConfig(gpuPID, gpuIdx, config, totalGPUMem); err != nil {
						fmt.Fprintf(os.Stderr, "[%s] Error applying controls to PID %d GPU %d (container %s): %v\n",
							ts(), gpuPID, gpuIdx, c.Name, err)
						allOK = false
					} else {
						memLimit := config.MemoryLimitForDevice(gpuIdx, totalGPUMem)
						fmt.Printf("[%s] Applied controls to PID %d GPU %d (container: %s) — memory=%s priority=%d freeze=%v\n",
							ts(), gpuPID, gpuIdx, c.Name,
							gvm.FormatMemoryValue(memLimit), config.ComputePriority, config.ComputeFreeze)
					}
				}

				if allOK {
					state.appliedPIDs[gpuPID] = true
				}
			}
		}

		// Cleanup stale containers
		for id := range containerStates {
			if !activeIDs[id] {
				delete(containerStates, id)
			}
		}

		time.Sleep(pollInterval)
	}
}

func getRunningContainers() ([]containerInfo, error) {
	cmd := exec.Command("docker", "ps", "-q")
	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	raw := strings.TrimSpace(string(output))
	if raw == "" {
		return nil, nil
	}

	containerIDs := strings.Split(raw, "\n")
	var containers []containerInfo

	for _, id := range containerIDs {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}

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

		envVars := make(map[string]string)
		for _, env := range data.Config.Env {
			parts := strings.SplitN(env, "=", 2)
			if len(parts) == 2 {
				envVars[parts[0]] = parts[1]
			}
		}

		containers = append(containers, containerInfo{
			ID:      data.ID,
			PID:     data.State.Pid,
			Name:    strings.TrimPrefix(data.Name, "/"),
			EnvVars: envVars,
		})
	}

	return containers, nil
}

func ts() string {
	return time.Now().Format("15:04:05")
}

func describeMemoryConfig(config *gvm.Config) string {
	parts := []string{}
	if config.MemoryLimit > 0 {
		parts = append(parts, fmt.Sprintf("global=%s", gvm.FormatMemoryValue(config.MemoryLimit)))
	}
	for idx, limit := range config.PerDeviceMemoryLimit {
		parts = append(parts, fmt.Sprintf("gpu%d=%s", idx, gvm.FormatMemoryValue(limit)))
	}
	if config.MemoryPercentage > 0 {
		parts = append(parts, fmt.Sprintf("percentage=%d%%", config.MemoryPercentage))
	}
	if len(parts) == 0 {
		return "unlimited"
	}
	return strings.Join(parts, ", ")
}
