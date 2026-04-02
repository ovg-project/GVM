package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/ovg-project/gvm-docker/pkg/gvm"
)

var runcPath string

func init() {
	// Ensure we have root privileges
	if syscall.Geteuid() != 0 {
		fmt.Fprintln(os.Stderr, "gvm-runc: must run as root")
		os.Exit(1)
	}
}

func main() {
	// Special mode: apply GVM controls as a background process.
	// Invoked as: gvm-runc --gvm-apply <containerID> <envJSON>
	if len(os.Args) >= 4 && os.Args[1] == "--gvm-apply" {
		runApplyMode(os.Args[2], os.Args[3])
		return
	}

	runcPath = findRunc()

	args := os.Args[1:]

	// Always log to file for debugging
	logf := newLogger()

	logf("gvm-runc invoked: args=%v", args)

	// Parse the runc command, bundle path, and container ID from args.
	// Docker typically calls: gvm-runc [global-flags] create --bundle <path> <container-id>
	var command, bundlePath, containerID string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--bundle" && i+1 < len(args) {
			bundlePath = args[i+1]
			i++ // skip next
		} else if arg == "create" || arg == "start" || arg == "run" {
			command = arg
		} else if !strings.HasPrefix(arg, "-") && command != "" {
			// First non-flag arg after the command is the container ID
			containerID = arg
		}
	}

	logf("  command=%s bundle=%s containerID=%s", command, bundlePath, containerID)

	// Read container env vars from the OCI bundle's config.json
	var containerEnv map[string]string
	if bundlePath != "" {
		containerEnv = readBundleEnv(bundlePath, logf)
	}

	// Execute the real runc
	cmd := exec.Command(runcPath, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			os.Exit(exitErr.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "gvm-runc: failed to run runc: %v\n", err)
		os.Exit(1)
	}

	// After a successful create, fork a background process to apply GVM controls.
	// We can't use a goroutine because main() returns and kills the goroutine.
	if command == "create" && containerID != "" && len(containerEnv) > 0 {
		envJSON, _ := json.Marshal(containerEnv)
		child := exec.Command(os.Args[0], "--gvm-apply", containerID, string(envJSON))
		child.Stdout = nil
		child.Stderr = nil
		if err := child.Start(); err != nil {
			logf("ERROR: failed to fork apply process: %v", err)
		} else {
			// Detach — don't wait for the child
			child.Process.Release()
			logf("Forked apply process (PID %d) for container %s", child.Process.Pid, containerID)
		}
	}
}

func newLogger() func(string, ...interface{}) {
	logFile, _ := os.OpenFile("/tmp/gvm-runc.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	return func(format string, a ...interface{}) {
		msg := fmt.Sprintf("[%s] %s\n", time.Now().Format(time.RFC3339), fmt.Sprintf(format, a...))
		if logFile != nil {
			fmt.Fprint(logFile, msg)
		}
	}
}

// runApplyMode is the entry point for the background apply process.
func runApplyMode(containerID, envJSON string) {
	logf := newLogger()
	logf("Apply process started for container %s", containerID)

	var envVars map[string]string
	if err := json.Unmarshal([]byte(envJSON), &envVars); err != nil {
		logf("ERROR: failed to parse env JSON: %v", err)
		return
	}

	applyControls(containerID, envVars, logf)
}

// readBundleEnv reads the container's environment variables from the OCI bundle config.json.
func readBundleEnv(bundlePath string, logf func(string, ...interface{})) map[string]string {
	configPath := filepath.Join(bundlePath, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		logf("WARNING: could not read bundle config %s: %v", configPath, err)
		return nil
	}

	var ociConfig struct {
		Process struct {
			Env []string `json:"env"`
		} `json:"process"`
	}
	if err := json.Unmarshal(data, &ociConfig); err != nil {
		logf("WARNING: could not parse bundle config %s: %v", configPath, err)
		return nil
	}

	envVars := make(map[string]string)
	for _, env := range ociConfig.Process.Env {
		parts := strings.SplitN(env, "=", 2)
		if len(parts) == 2 && strings.HasPrefix(parts[0], "GVM_") {
			envVars[parts[0]] = parts[1]
		}
	}

	if len(envVars) > 0 {
		logf("  Found GVM env vars in bundle: %v", envVars)
	}

	return envVars
}

func findRunc() string {
	// Avoid finding ourselves — skip /usr/local/bin/gvm-runc
	paths := []string{
		"/usr/bin/runc",
		"/usr/sbin/runc",
		"/run/current-system/sw/bin/runc",
	}
	for _, path := range paths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	if path, err := exec.LookPath("runc"); err == nil {
		// Make sure we didn't find ourselves
		if !strings.Contains(path, "gvm-runc") {
			return path
		}
	}
	return "/usr/bin/runc"
}

func applyControls(containerID string, envVars map[string]string, logf func(string, ...interface{})) {
	config, err := gvm.ConfigFromEnv(envVars)
	if err != nil {
		logf("ERROR parsing config for %s: %v", containerID, err)
		return
	}

	if !config.Enabled || !config.HasAnyControl() {
		logf("No GVM controls to apply for container %s", containerID)
		return
	}

	logf("Starting GVM control application for container %s", containerID)
	logf("  Memory limit: %s, Priority: %d, Freeze: %v",
		gvm.FormatMemoryValue(config.MemoryLimit), config.ComputePriority, config.ComputeFreeze)

	// Get container PID via docker inspect
	containerPID, err := getContainerPID(containerID)
	if err != nil {
		logf("ERROR: could not get container PID: %v", err)
		return
	}
	logf("Container PID: %d", containerPID)

	// Retry finding GPU processes for up to 60 seconds
	var pids []int
	maxRetries := 30
	for i := 0; i < maxRetries; i++ {
		time.Sleep(2 * time.Second)

		pids, err = gvm.FindGPUPIDs(containerPID)
		if err != nil {
			if i%5 == 0 {
				logf("Attempt %d/%d: error finding GPU processes for %s: %v", i+1, maxRetries, containerID, err)
			}
			continue
		}

		if len(pids) > 0 {
			logf("Found %d GPU process(es) after %d seconds: %v", len(pids), (i+1)*2, pids)
			break
		}

		if i%5 == 0 {
			logf("Still waiting for GPU processes (attempt %d/%d)", i+1, maxRetries)
		}
	}

	if len(pids) == 0 {
		logf("WARNING: No GPU processes found after %d seconds for %s", maxRetries*2, containerID)
		return
	}

	// Query GPU memory for percentage-based limits
	gpuMemory, memErr := gvm.QueryGPUTotalMemory()
	if memErr != nil {
		logf("Warning: could not query GPU memory: %v (percentage limits won't work)", memErr)
		gpuMemory = make(map[int]int64)
	}

	// Apply controls
	for _, pid := range pids {
		gpuIndices, err := gvm.ListGPUIndices(pid)
		if err != nil {
			logf("ERROR listing GPU indices for PID %d: %v", pid, err)
			continue
		}

		for _, gpuIdx := range gpuIndices {
			totalMem := gpuMemory[gpuIdx]
			if err := gvm.ApplyConfig(pid, gpuIdx, config, totalMem); err != nil {
				logf("ERROR applying controls to PID %d GPU %d: %v", pid, gpuIdx, err)
			} else {
				memLimit := config.MemoryLimitForDevice(gpuIdx, totalMem)
				logf("SUCCESS: Applied to PID %d GPU %d — memory=%s priority=%d freeze=%v",
					pid, gpuIdx, gvm.FormatMemoryValue(memLimit), config.ComputePriority, config.ComputeFreeze)
			}
		}
	}
}

// getContainerPID uses docker inspect to get the main PID of a container.
func getContainerPID(containerID string) (int, error) {
	cmd := exec.Command("docker", "inspect", "--format", "{{.State.Pid}}", containerID)
	output, err := cmd.Output()
	if err != nil {
		return 0, fmt.Errorf("docker inspect failed: %w", err)
	}

	pidStr := strings.TrimSpace(string(output))
	var pid int
	if _, err := fmt.Sscanf(pidStr, "%d", &pid); err != nil {
		return 0, fmt.Errorf("parse PID %q: %w", pidStr, err)
	}
	if pid == 0 {
		return 0, fmt.Errorf("container not running")
	}
	return pid, nil
}
