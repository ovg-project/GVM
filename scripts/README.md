## Setup

```bash
./download_pkgs.sh

./install_cuda.sh
./install_nv_driver.sh
```

## Compile

```bash
./compile_modules.sh
```

## Install/uninstall kernel modules

```bash
./deploy_modules.sh  # if first time deploy kernel modules
./redeploy_modules.sh  # uninstall previous kernel modules and re-deploy them
./uninstall_modules.sh  # uninstall previous kernel modules
```
