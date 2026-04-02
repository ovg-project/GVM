package gvm

import (
	"testing"
)

func TestParseMemoryValue(t *testing.T) {
	tests := []struct {
		input    string
		expected int64
		wantErr  bool
	}{
		// Raw bytes
		{"0", 0, false},
		{"1000", 1000, false},
		{"6000000000", 6000000000, false},

		// Byte suffix
		{"100b", 100, false},
		{"100B", 100, false},

		// Kilobytes
		{"1k", 1000, false},
		{"1K", 1000, false},
		{"1500k", 1500000, false},

		// Megabytes
		{"1m", 1000000, false},
		{"1M", 1000000, false},
		{"6000m", 6000000000, false},
		{"16000m", 16000000000, false},

		// Gigabytes
		{"1g", 1000000000, false},
		{"1G", 1000000000, false},
		{"6g", 6000000000, false},
		{"24g", 24000000000, false},

		// Terabytes
		{"1t", 1000000000000, false},
		{"1T", 1000000000000, false},

		// Decimal values
		{"1.5g", 1500000000, false},
		{"0.5g", 500000000, false},
		{"2.5m", 2500000, false},

		// Whitespace
		{" 6g ", 6000000000, false},
		{" 1000 ", 1000, false},

		// Errors
		{"", 0, true},
		{"g", 0, true},
		{"-1g", 0, true},
		{"abc", 0, true},
		{"abc-g", 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result, err := ParseMemoryValue(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Errorf("ParseMemoryValue(%q) expected error, got %d", tt.input, result)
				}
				return
			}
			if err != nil {
				t.Errorf("ParseMemoryValue(%q) unexpected error: %v", tt.input, err)
				return
			}
			if result != tt.expected {
				t.Errorf("ParseMemoryValue(%q) = %d, want %d", tt.input, result, tt.expected)
			}
		})
	}
}

func TestFormatMemoryValue(t *testing.T) {
	tests := []struct {
		input    int64
		expected string
	}{
		{0, "0"},
		{500, "500"},
		{1000, "1k"},
		{1500000, "1500k"},
		{1000000, "1m"},
		{6000000000, "6g"},
		{1000000000000, "1t"},
		{1500000000, "1500m"},
		{1234, "1234"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			result := FormatMemoryValue(tt.input)
			if result != tt.expected {
				t.Errorf("FormatMemoryValue(%d) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestConfigFromEnv(t *testing.T) {
	t.Run("empty env", func(t *testing.T) {
		config, err := ConfigFromEnv(map[string]string{})
		if err != nil {
			t.Fatal(err)
		}
		if !config.Enabled {
			t.Error("expected Enabled=true by default")
		}
		if config.ComputePriority != DefaultPriority {
			t.Errorf("expected default priority %d, got %d", DefaultPriority, config.ComputePriority)
		}
		if config.HasAnyControl() {
			t.Error("expected HasAnyControl()=false with no env vars")
		}
	})

	t.Run("basic controls", func(t *testing.T) {
		config, err := ConfigFromEnv(map[string]string{
			"GVM_MEMORY_LIMIT":     "6g",
			"GVM_COMPUTE_PRIORITY": "2",
		})
		if err != nil {
			t.Fatal(err)
		}
		if config.MemoryLimit != 6000000000 {
			t.Errorf("MemoryLimit = %d, want 6000000000", config.MemoryLimit)
		}
		if config.ComputePriority != 2 {
			t.Errorf("ComputePriority = %d, want 2", config.ComputePriority)
		}
		if !config.HasAnyControl() {
			t.Error("expected HasAnyControl()=true")
		}
	})

	t.Run("per-device limits", func(t *testing.T) {
		config, err := ConfigFromEnv(map[string]string{
			"GVM_MEMORY_LIMIT":   "4g",
			"GVM_MEMORY_LIMIT_0": "8g",
			"GVM_MEMORY_LIMIT_1": "6g",
		})
		if err != nil {
			t.Fatal(err)
		}
		if config.MemoryLimit != 4000000000 {
			t.Errorf("MemoryLimit = %d, want 4000000000", config.MemoryLimit)
		}
		if config.PerDeviceMemoryLimit[0] != 8000000000 {
			t.Errorf("PerDeviceMemoryLimit[0] = %d, want 8000000000", config.PerDeviceMemoryLimit[0])
		}
		if config.PerDeviceMemoryLimit[1] != 6000000000 {
			t.Errorf("PerDeviceMemoryLimit[1] = %d, want 6000000000", config.PerDeviceMemoryLimit[1])
		}
	})

	t.Run("percentage", func(t *testing.T) {
		config, err := ConfigFromEnv(map[string]string{
			"GVM_MEMORY_PERCENTAGE": "50",
		})
		if err != nil {
			t.Fatal(err)
		}
		if config.MemoryPercentage != 50 {
			t.Errorf("MemoryPercentage = %d, want 50", config.MemoryPercentage)
		}
	})

	t.Run("freeze", func(t *testing.T) {
		config, err := ConfigFromEnv(map[string]string{
			"GVM_COMPUTE_FREEZE": "true",
		})
		if err != nil {
			t.Fatal(err)
		}
		if !config.ComputeFreeze {
			t.Error("expected ComputeFreeze=true")
		}
	})

	t.Run("disabled", func(t *testing.T) {
		config, err := ConfigFromEnv(map[string]string{
			"GVM_ENABLED":      "false",
			"GVM_MEMORY_LIMIT": "6g",
		})
		if err != nil {
			t.Fatal(err)
		}
		if config.Enabled {
			t.Error("expected Enabled=false")
		}
		// When disabled, memory limit should not be parsed
		if config.MemoryLimit != 0 {
			t.Errorf("MemoryLimit = %d, want 0 (disabled)", config.MemoryLimit)
		}
	})

	t.Run("debug", func(t *testing.T) {
		config, err := ConfigFromEnv(map[string]string{
			"GVM_DEBUG": "true",
		})
		if err != nil {
			t.Fatal(err)
		}
		if !config.Debug {
			t.Error("expected Debug=true")
		}
	})

	t.Run("invalid priority", func(t *testing.T) {
		_, err := ConfigFromEnv(map[string]string{
			"GVM_COMPUTE_PRIORITY": "20",
		})
		if err == nil {
			t.Error("expected error for priority 20")
		}
	})

	t.Run("invalid percentage", func(t *testing.T) {
		_, err := ConfigFromEnv(map[string]string{
			"GVM_MEMORY_PERCENTAGE": "150",
		})
		if err == nil {
			t.Error("expected error for percentage 150")
		}
	})

	t.Run("invalid memory value", func(t *testing.T) {
		_, err := ConfigFromEnv(map[string]string{
			"GVM_MEMORY_LIMIT": "notanumber",
		})
		if err == nil {
			t.Error("expected error for invalid memory value")
		}
	})
}

func TestMemoryLimitForDevice(t *testing.T) {
	config := &Config{
		MemoryLimit: 4000000000,
		PerDeviceMemoryLimit: map[int]int64{
			0: 8000000000,
		},
		MemoryPercentage: 50,
	}

	t.Run("per-device override", func(t *testing.T) {
		limit := config.MemoryLimitForDevice(0, 24000000000)
		if limit != 8000000000 {
			t.Errorf("GPU 0 limit = %d, want 8000000000 (per-device)", limit)
		}
	})

	t.Run("global fallback", func(t *testing.T) {
		limit := config.MemoryLimitForDevice(1, 24000000000)
		if limit != 4000000000 {
			t.Errorf("GPU 1 limit = %d, want 4000000000 (global)", limit)
		}
	})

	t.Run("percentage fallback", func(t *testing.T) {
		pctConfig := &Config{
			MemoryPercentage:     50,
			PerDeviceMemoryLimit: make(map[int]int64),
		}
		limit := pctConfig.MemoryLimitForDevice(0, 24000000000)
		if limit != 12000000000 {
			t.Errorf("GPU 0 pct limit = %d, want 12000000000 (50%% of 24g)", limit)
		}
	})

	t.Run("no limit", func(t *testing.T) {
		noConfig := &Config{
			PerDeviceMemoryLimit: make(map[int]int64),
		}
		limit := noConfig.MemoryLimitForDevice(0, 24000000000)
		if limit != 0 {
			t.Errorf("GPU 0 limit = %d, want 0 (unlimited)", limit)
		}
	})
}
