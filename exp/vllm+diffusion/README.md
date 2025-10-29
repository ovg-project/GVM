## Usage

### Prerequisites

Must have the corresponding scheduler (`GVM` or `xsched`) installed.

### Prerequisites for running the script

```bash
# Install uv (and python)
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /path/to/gvm-dev
uv venv
source .venv/bin/activate
uv pip install torch diffusers transformers protobuf accelerate sentencepiece vllm
```

GVM works for unmodified vLLM. 
For XSched, which requires a manually created CUDA stream instead of the default one, vLLM needs to be patched with a few lines of simple modifications.
Please refer to the push-button setup script: `scripts/setup.sh`.


### Prepare datasets

```shell
cd /path/to/gvm-dev/exp/vllm+diffusion
./download_dataset.sh
# Prepare vidprom dataset
python csv_to_prompts.py datasets/VidProM_unique_example.csv -o datasets/vidprom_prompts.txt
```

### Before running XSched (only XSched, no need for GVM)

```shell
./launch_xserver.sh
```

### Run

#### Diffusion inference

```shell
./start_diffusion.sh [gvm|xsched|none]
```

#### vLLM

```shell
# Might need to adjust --max-model-len for different accelerators.
# If seeing a vLLM OOM, try reducing the --max-model-len.
./start_vllm_server.sh [gvm|xsched|none]
# Waiting for vLLM server starts...
./start_vllm_client.sh
```

### After launching apps with GVM

```shell
./config_gvm.sh
```

To config each application's compute priority and memory limit. Check the script to adjust parameters.
