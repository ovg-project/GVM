#!/usr/bin/env bash
# Uninstall proprietary NVIDIA driver on Ubuntu (apt/.run/DKMS/manual)
# Usage: sudo bash uninstall_nvidia.sh [--enable-nouveau]

set -euo pipefail

ENABLE_NOUVEAU=0
[[ "${1:-}" == "--enable-nouveau" ]] && ENABLE_NOUVEAU=1

need_root() {
  if [[ $EUID -ne 0 ]]; then
    echo "Please run as root: sudo bash $0 [--enable-nouveau]" >&2
    exit 1
  fi
}

step() { echo -e "\n==> $*"; }

stop_services() {
  step "Stopping display manager and NVIDIA services"
  systemctl stop nvidia-persistenced 2>/dev/null || true
  # Try common DMs; ignore errors if not present
  for dm in display-manager gdm3 sddm lightdm; do
    systemctl stop "$dm" 2>/dev/null || true
  done
}

unload_modules() {
  step "Attempting to unload NVIDIA kernel modules..."
  # Try multiple times in case something is holding the device
  for i in {1..3}; do
    modprobe -r nvidia_drm nvidia_uvm nvidia_modeset nvidia 2>/dev/null || true
    if ! lsmod | egrep -q '(^nvidia| nvidia_|uvm|modeset)'; then
      echo "All NVIDIA modules unloaded."
      return 0
    fi
    echo "Some modules still loaded; checking and killing processes..."
    # Get PIDs of processes holding NVIDIA devices and kill them
    local pids
    pids=$(fuser /dev/nvidia* /dev/dri/* 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
      echo "Found processes using NVIDIA devices: $pids"
      echo "Killing processes..."
      kill -9 $pids 2>/dev/null || true
      sleep 1
    fi
    sleep 1
  done
  # One last attempt
  modprobe -r nvidia_drm nvidia_uvm nvidia_modeset nvidia 2>/dev/null || true
  if lsmod | egrep -q '(^nvidia| nvidia_|uvm|modeset)'; then
    echo "Warning: NVIDIA modules still loaded. Ensure no GUI/session is using them and re-run." >&2
  fi
}

runfile_uninstall() {
  step "Attempting NVIDIA .run uninstaller (if present)"
  if command -v nvidia-uninstall >/dev/null 2>&1; then
    nvidia-uninstall --silent || true
  elif [[ -x /usr/bin/nvidia-uninstall ]]; then
    /usr/bin/nvidia-uninstall --silent || true
  else
    echo "No .run uninstaller found; skipping."
  fi
}

dkms_remove() {
  step "Removing DKMS builds (if any)"
  if command -v dkms >/dev/null 2>&1; then
    dkms status | awk -F'[ ,:]+' '/nvidia/ {print $1,$2}' | while read -r module ver; do
      echo "dkms remove -m ${module} -v ${ver} --all"
      dkms remove -m "${module}" -v "${ver}" --all || true
    done
  else
    echo "DKMS not installed; skipping."
  fi
}

apt_purge() {
  step "Purging Ubuntu NVIDIA packages"
  export DEBIAN_FRONTEND=noninteractive
  # Purge broad patterns safely (ignore if not installed)
  apt-get -y purge \
    'nvidia-*' 'libnvidia-*' 'xserver-xorg-video-nvidia*' \
    'cuda-*' 'libcuda*' dkms 2>/dev/null || true

  # Clean reverse-deps and residual configs
  apt-get -y autoremove --purge 2>/dev/null || true
  apt-get -y autoclean 2>/dev/null || true
}

remove_leftovers() {
  step "Removing leftover kernel modules and extras for current kernel"
  local KREL
  KREL=$(uname -r)

  find "/lib/modules/${KREL}" -type f -name 'nvidia*.ko*' -print -delete 2>/dev/null || true
  find "/lib/modules/${KREL}" -type f -name 'nv*uvm*.ko*' -print -delete 2>/dev/null || true
  rm -rf "/lib/modules/${KREL}/extra/nvidia*" "/lib/modules/${KREL}/updates/dkms/nvidia*" 2>/dev/null || true

  step "Cleaning misc NVIDIA paths (safe to ignore if missing)"
  rm -rf /usr/lib/nvidia /usr/lib32/nvidia /usr/lib/x86_64-linux-gnu/nvidia 2>/dev/null || true
  rm -rf /usr/share/X11/xorg.conf.d/10-nvidia*.conf 2>/dev/null || true
  rm -f  /etc/X11/xorg.conf 2>/dev/null || true
}

ensure_build_tools() {
  step "Ensuring build tools and kernel headers are installed"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq 2>/dev/null || true
  apt-get install -y build-essential "linux-headers-$(uname -r)" 2>/dev/null || true
}

blacklist_nouveau() {
  if [[ $ENABLE_NOUVEAU -eq 0 ]]; then
    step "Blacklisting nouveau driver"
    echo -e "blacklist nouveau\noptions nouveau modeset=0" > /etc/modprobe.d/blacklist-nouveau.conf
    echo "Nouveau driver blacklisted in /etc/modprobe.d/blacklist-nouveau.conf"
  else
    step "Removing nouveau blacklist (if any)"
    rm -f /etc/modprobe.d/blacklist-nouveau.conf 2>/dev/null || true
  fi
}

regen_initramfs() {
  step "Regenerating module deps and initramfs"
  depmod -a
  if command -v update-initramfs >/dev/null 2>&1; then
    update-initramfs -u
  else
    echo "update-initramfs not found (non-Ubuntu system?). Skipping."
  fi
}

final_note() {
  step "Done."
  echo "Recommended: reboot now to complete removal."
  echo "After reboot, verify:"
  echo "  lsmod | egrep '(^nvidia| nvidia_|uvm|modeset)'  # should show nothing"
  if [[ $ENABLE_NOUVEAU -eq 1 ]]; then
    echo "  lsmod | grep nouveau  # should show nouveau if it loaded"
  fi
}

main() {
  need_root
  stop_services
  unload_modules
  runfile_uninstall
  dkms_remove
  apt_purge
  remove_leftovers
  blacklist_nouveau
  regen_initramfs
  final_note
  # ensure_build_tools
}

main "$@"