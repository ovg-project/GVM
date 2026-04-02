# Required Fixes for Submodule Scripts

This document outlines the changes needed in the submodule repositories (`gvm-nvidia-driver-modules` and `gvm-cuda-driver`) to address issues discovered during setup.

## gvm-nvidia-driver-modules

### Fix 1: Update `install_cuda.sh` to install toolkit only

**File:** `scripts/install_cuda.sh`

**Current code:**
```bash
#!/bin/bash
sudo sh cuda_12.9.1_575.57.08_linux.run --check
sudo sh cuda_12.9.1_575.57.08_linux.run --silent --driver --toolkit --override
```

**Issue:** The bundled driver (575.57.08) fails to compile on kernel 6.8+ due to removed kernel APIs.

**Proposed fix:**
```bash
#!/bin/bash
sudo sh cuda_12.9.1_575.57.08_linux.run --check
# Install toolkit only, without bundled driver (which is incompatible with kernel 6.8+)
sudo sh cuda_12.9.1_575.57.08_linux.run --silent --toolkit --override --no-drm
echo "CUDA toolkit installed. Run install_nv_driver.sh next to install the GVM driver."
```

### Fix 2: Add kernel version check to `compile_modules.sh`

**File:** `scripts/compile_modules.sh`

**Proposed addition at the beginning:**
```bash
#!/bin/bash

# Check kernel version compatibility
KERNEL_VERSION=$(uname -r | cut -d. -f1,2)
KERNEL_MAJOR=$(echo $KERNEL_VERSION | cut -d. -f1)
KERNEL_MINOR=$(echo $KERNEL_VERSION | cut -d. -f2)

if [ "$KERNEL_MAJOR" -gt 6 ] || ([ "$KERNEL_MAJOR" -eq 6 ] && [ "$KERNEL_MINOR" -gt 8 ]); then
    echo "WARNING: Kernel version $KERNEL_VERSION detected."
    echo "GVM kernel modules are tested on kernel 6.8 and may not compile on newer kernels."
    echo "If compilation fails, consider downgrading to kernel 6.8."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Rest of the existing script...
```

### Fix 3: Add PATH setup reminder to README

**File:** `README.md`

Add a note after the CUDA installation step:
```markdown
**Note:** After installing CUDA, add it to your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

Add this to your `~/.bashrc` to make it permanent:
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
```
```

## gvm-cuda-driver

### Fix 1: Add PATH check to Makefile or build script

**File:** `Makefile` or create a new `build.sh` wrapper

**Proposed addition:**
```bash
#!/bin/bash
# build.sh - Wrapper script to ensure nvcc is in PATH

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found in PATH"
    echo "Please add CUDA to your PATH:"
    echo "  export PATH=/usr/local/cuda/bin:\$PATH"
    echo ""
    echo "Or add it permanently to ~/.bashrc:"
    echo "  echo 'export PATH=/usr/local/cuda/bin:\$PATH' >> ~/.bashrc"
    exit 1
fi

# Run the actual make command
make "$@"
```

Then update README to suggest using `./build.sh` instead of `make` directly, or add the PATH check directly in the Makefile.

## Summary of Changes

These fixes address the following issues:
1. **Issue 2** - CUDA bundled driver incompatibility → Fixed by `install_cuda.sh` change
2. **Issue 3** - Kernel version incompatibility → Fixed by adding version check in `compile_modules.sh`
3. **Issue 4** - Missing nvcc in PATH → Fixed by adding PATH checks and documentation

The other issues are already addressed in the main GVM repository:
- **Issue 1** - SSH URLs → Fixed by changing `.gitmodules` to HTTPS
- **Issue 5** - HuggingFace auth → Documented in README troubleshooting
- **Issue 6** - Diffusion output directory → Fixed by updating `setup` script
