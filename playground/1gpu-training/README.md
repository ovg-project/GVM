## Setup

Install Llama-factory:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
# Follow llama-factory instructions to install
# ...

cp llama3_full_sft.yaml LLaMA-Factory/
cp llama3_lora_sft.yaml LLaMA-Factory/
```

## Run

```bash
cd Llama-Factory
llamafactory-cli train llama3_[full|lora]_sft.yaml
```

### Run with LD_PRELOAD

```bash
pushd <path to gvm-dev>/csrc/custom_cuda_lib/
make -j
popd

LD_PRELOAD=<path to gvm-dev>/csrc/custom_cuda_lib/libcustom_cuda.so  llamafactory-cli train llama3_[full|lora]_sft.yaml
```