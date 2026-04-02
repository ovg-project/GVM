package gvm

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Config holds all GVM control parameters for a container.
type Config struct {
	// Global memory limit in bytes. 0 means unlimited.
	MemoryLimit int64

	// Per-device memory limits in bytes, keyed by GPU index.
	// Overrides MemoryLimit for the specified device.
	PerDeviceMemoryLimit map[int]int64

	// Memory limit as percentage of total GPU memory (1-100).
	// Only used if MemoryLimit is not set.
	MemoryPercentage int

	// Compute priority (0-15). 0 = highest priority, 15 = lowest.
	ComputePriority int

	// Whether to freeze GPU compute for this container.
	ComputeFreeze bool

	// Whether GVM controls are enabled for this container.
	Enabled bool

	// Whether debug logging is enabled.
	Debug bool
}

const DefaultPriority = 8

// ConfigFromEnv parses GVM configuration from a map of environment variables.
// This is used by the daemon which reads env vars from docker inspect.
func ConfigFromEnv(envVars map[string]string) (*Config, error) {
	config := &Config{
		ComputePriority:      DefaultPriority,
		Enabled:              true,
		PerDeviceMemoryLimit: make(map[int]int64),
	}

	// GVM_ENABLED
	if val, ok := envVars["GVM_ENABLED"]; ok {
		config.Enabled = strings.ToLower(val) != "false"
	}

	if !config.Enabled {
		return config, nil
	}

	// GVM_DEBUG
	config.Debug = strings.ToLower(envVars["GVM_DEBUG"]) == "true"

	// GVM_MEMORY_LIMIT (global)
	if val, ok := envVars["GVM_MEMORY_LIMIT"]; ok && val != "" {
		bytes, err := ParseMemoryValue(val)
		if err != nil {
			return nil, fmt.Errorf("invalid GVM_MEMORY_LIMIT %q: %w", val, err)
		}
		config.MemoryLimit = bytes
	}

	// GVM_MEMORY_LIMIT_<N> (per-device)
	for key, val := range envVars {
		if !strings.HasPrefix(key, "GVM_MEMORY_LIMIT_") {
			continue
		}
		suffix := strings.TrimPrefix(key, "GVM_MEMORY_LIMIT_")
		gpuIdx, err := strconv.Atoi(suffix)
		if err != nil {
			continue // not a numeric suffix, skip
		}
		bytes, err := ParseMemoryValue(val)
		if err != nil {
			return nil, fmt.Errorf("invalid %s %q: %w", key, val, err)
		}
		config.PerDeviceMemoryLimit[gpuIdx] = bytes
	}

	// GVM_MEMORY_PERCENTAGE
	if val, ok := envVars["GVM_MEMORY_PERCENTAGE"]; ok && val != "" {
		pct, err := strconv.Atoi(val)
		if err != nil || pct < 1 || pct > 100 {
			return nil, fmt.Errorf("invalid GVM_MEMORY_PERCENTAGE %q: must be 1-100", val)
		}
		config.MemoryPercentage = pct
	}

	// GVM_COMPUTE_PRIORITY
	if val, ok := envVars["GVM_COMPUTE_PRIORITY"]; ok && val != "" {
		p, err := strconv.Atoi(val)
		if err != nil || p < 0 || p > 15 {
			return nil, fmt.Errorf("invalid GVM_COMPUTE_PRIORITY %q: must be 0-15", val)
		}
		config.ComputePriority = p
	}

	// GVM_COMPUTE_FREEZE
	if val, ok := envVars["GVM_COMPUTE_FREEZE"]; ok {
		config.ComputeFreeze = strings.ToLower(val) == "true"
	}

	return config, nil
}

// ConfigFromOSEnv parses GVM configuration from the current process environment.
// This is used by the runc wrapper which inherits container env vars.
func ConfigFromOSEnv() (*Config, error) {
	envVars := make(map[string]string)
	for _, env := range os.Environ() {
		parts := strings.SplitN(env, "=", 2)
		if len(parts) == 2 && strings.HasPrefix(parts[0], "GVM_") {
			envVars[parts[0]] = parts[1]
		}
	}
	return ConfigFromEnv(envVars)
}

// HasAnyControl returns true if this config has any GVM control set
// (i.e., there's something to apply beyond defaults).
func (c *Config) HasAnyControl() bool {
	if !c.Enabled {
		return false
	}
	return c.MemoryLimit > 0 ||
		len(c.PerDeviceMemoryLimit) > 0 ||
		c.MemoryPercentage > 0 ||
		c.ComputePriority != DefaultPriority ||
		c.ComputeFreeze
}

// MemoryLimitForDevice returns the effective memory limit in bytes for a
// specific GPU device index. Returns 0 if no limit is configured.
//
// Priority: per-device limit > global limit > percentage-based limit.
// For percentage-based, totalGPUMemory must be provided (in bytes).
func (c *Config) MemoryLimitForDevice(gpuIndex int, totalGPUMemory int64) int64 {
	// Per-device override takes highest priority
	if limit, ok := c.PerDeviceMemoryLimit[gpuIndex]; ok {
		return limit
	}

	// Global memory limit
	if c.MemoryLimit > 0 {
		return c.MemoryLimit
	}

	// Percentage-based
	if c.MemoryPercentage > 0 && totalGPUMemory > 0 {
		return totalGPUMemory * int64(c.MemoryPercentage) / 100
	}

	return 0
}

// ParseMemoryValue parses a human-readable memory string into bytes.
// Supports: raw bytes ("6000000000"), or suffixed values ("6g", "6000m", "1500k", "100b").
// Uses SI units (powers of 10): 1g = 1,000,000,000 bytes.
func ParseMemoryValue(s string) (int64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, fmt.Errorf("empty memory value")
	}

	// Check for unit suffix
	lastChar := strings.ToLower(s[len(s)-1:])
	multiplier := int64(1)
	numStr := s

	switch lastChar {
	case "t":
		multiplier = 1_000_000_000_000
		numStr = s[:len(s)-1]
	case "g":
		multiplier = 1_000_000_000
		numStr = s[:len(s)-1]
	case "m":
		multiplier = 1_000_000
		numStr = s[:len(s)-1]
	case "k":
		multiplier = 1_000
		numStr = s[:len(s)-1]
	case "b":
		multiplier = 1
		numStr = s[:len(s)-1]
	default:
		// No suffix — treat as raw bytes
		multiplier = 1
		numStr = s
	}

	numStr = strings.TrimSpace(numStr)
	if numStr == "" {
		return 0, fmt.Errorf("no numeric value in %q", s)
	}

	// Support decimal values like "1.5g"
	if strings.Contains(numStr, ".") {
		val, err := strconv.ParseFloat(numStr, 64)
		if err != nil {
			return 0, fmt.Errorf("invalid number %q: %w", numStr, err)
		}
		if val < 0 {
			return 0, fmt.Errorf("negative memory value: %s", s)
		}
		return int64(val * float64(multiplier)), nil
	}

	val, err := strconv.ParseInt(numStr, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid number %q: %w", numStr, err)
	}
	if val < 0 {
		return 0, fmt.Errorf("negative memory value: %s", s)
	}

	return val * multiplier, nil
}

// FormatMemoryValue formats a byte count into a human-readable string.
func FormatMemoryValue(bytes int64) string {
	if bytes == 0 {
		return "0"
	}
	if bytes >= 1_000_000_000_000 && bytes%1_000_000_000_000 == 0 {
		return fmt.Sprintf("%dt", bytes/1_000_000_000_000)
	}
	if bytes >= 1_000_000_000 && bytes%1_000_000_000 == 0 {
		return fmt.Sprintf("%dg", bytes/1_000_000_000)
	}
	if bytes >= 1_000_000 && bytes%1_000_000 == 0 {
		return fmt.Sprintf("%dm", bytes/1_000_000)
	}
	if bytes >= 1_000 && bytes%1_000 == 0 {
		return fmt.Sprintf("%dk", bytes/1_000)
	}
	return fmt.Sprintf("%d", bytes)
}
