# Local Installation Guide for Kimi-K2-Instruct & GPT-OSS Models

This README walks you through the steps to download, install, and run the **Kimi-K2-Instruct** model and the **GPT-OSS (gpt-oss-120b / gpt-oss-20b)** models from Hugging Face on your local machine.

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Hugging Face License Acceptance](#hugging-face-license-acceptance)  
3. [Environment Setup](#environment-setup)  
4. [Installing Kimi-K2-Instruct](#installing-kimi-k2-instruct)  
5. [Installing GPT-OSS Models](#installing-gpt-oss-models)  
6. [Quick Start Examples](#quick-start-examples)  


---

## Prerequisites

- **Operating System**: Linux (Ubuntu 20.04+ recommended) or Windows 10/11 + WSL2  
- **CUDA-enabled GPU** with ≥24 GB VRAM for large models (e.g. A100, 3090/4090, H100)  
- **Python**: ≥ 3.8 (3.10+ recommended)  
- **Conda** (or virtualenv) for managing environments  
- **Git & Git LFS** for cloning large model weights  

---

## Hugging Face License Acceptance

Many community models require you to **Accept Model License** before downloading:

1. Visit the model page on Hugging Face, e.g.  
   - Kimi-K2-Instruct: `https://huggingface.co/moonshotai/Kimi-K2-Instruct`  
   - GPT-OSS-120B: `https://huggingface.co/openai/gpt-oss-120b`  
2. Click **“Accept terms”** or **“I agree to the model license”**.  
3. Log in with your Hugging Face account if prompted.  

Once accepted, your account is authorized to pull the weights via `transformers` or `git clone`.  

You can find more detail in the tutorial:  
https://huggingface.co/blog/proflead/hugging-face-tutorial 

---

## Environment Setup

1. **Install Git LFS**  
    ```bash
    sudo apt update
    sudo apt install git-lfs -y
    git lfs install
    ```

2. **Create & activate Conda env**  
    ```bash
    conda create -n llm-env python=3.10 -y
    conda activate llm-env
    ```

3. **Install core Python packages**  
    ```bash
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate huggingface_hub
    ```

---

## Installing Kimi-K2-Instruct  
Model on HuggingFace: [Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct)  

1. **Log in via CLI** (to cache your token):  
    ```bash
    huggingface-cli login
    ```

2. **Clone the repository**  
    ```bash
    git clone https://huggingface.co/moonshotai/Kimi-K2-Instruct
    cd Kimi-K2-Instruct
    ```

3. **Run a quick inference test**  
    Create `run_kimi.py`:
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_dir = "./"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).cuda()

    inputs = tokenizer("Explain the basics of quantum entanglement.", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=150)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    ```

    ```bash
    python run_kimi.py
    ```

---

## Installing GPT-OSS Models
Model on HuggingFace: [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b)  
Model on HuggingFace: [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b)  

<!--0. **(Optional) Upgrade Transformers**  -->
<!--    If you see `KeyError: 'gpt_oss'`, install the latest mainline:  -->
<!--    ```bash -->
<!--    pip uninstall -y transformers tokenizers -->
<!--    pip install git+https://github.com/huggingface/transformers.git -->
<!--    ``` -->

1. **Run inference via Transformers**  `oai_oss.py`   
    ```python
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model="openai/gpt-oss-120b",
        torch_dtype="auto",
        device_map="auto",
    )
    result = pipe("Briefly describe LLM quantization.", max_new_tokens=100)
    print(result[0]["generated_text"])
    ```
    ```bash
    python oai_oss.py
    ```

<!--2. **(Alternatively) Inference via vLLM**  -->
<!--    ```bash -->
<!--    pip install vllm --pre -->
<!--    ``` -->
<!--    ```python -->
<!--    from vllm import LLM -->
<!--    engine = LLM.from_pretrained("openai/gpt-oss-120b") -->
<!--    print(engine.generate("Hello, world!")) -->
<!--    ```  -->

---

## Quick Start Examples

| Model               | Command/Script               | Notes                         |
|---------------------|------------------------------|-------------------------------|
| **Kimi-K2-Instruct**| `python run_kimi.py`         | Uses `trust_remote_code` flag |
| **GPT-OSS-120B**    | Inline Python via `pipeline` | May require nightly `transformers` |
| **GPT-OSS-20B**     | Same as 120B, just swap name | Lower VRAM needs (~16 GB)     |

<!-- --- -->

<!-- ## Optional: Service Deployment with vLLM -->

<!-- Expose an OpenAI-compatible HTTP API: -->

<!-- ```bash -->
<!-- pip install vllm --pre -->
<!-- python3 -m vllm.entrypoints.openai.api_server \ -->
<!--    --model moonshotai/Kimi-K2-Instruct \ -->
<!--    --host 0.0.0.0 --port 8000 -->
