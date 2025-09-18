<div align="center">
    <h1 align="center">Think, Verbalize, then Speak</h1>
</div>

<div align="center">

### Bridging Complex Thoughts and Comprehensible Speech

[Sang Hoon Woo*](), [Sehun Lee*](https://yhytoto12.github.io), [Kang-wook Kim](), [Gunhee Kim](https://vision.snu.ac.kr/gunhee)

</div>

<div align="center">

[[ 🌐 Project Page ](https://yhytoto12.github.io/TVS-ReVerT)] [[ 📄 Paper ]()] [[ 🤗 Datasets ](#-datasets)] [[ 🤖️ Models ](#️-scripts)]

</div>

## 💥 News

- `2025.09.19` 🚀 We release the [arXiv paper]().
- `2025.09.19` 🔥 We release the training code, datasets, models and interactive demo.
- `2025.09.18` 🎉 Our paper got accepted to **EMNLP 2025**!

## 👀 Introduction

Recent spoken dialogue systems leverage large language models (LLMs) for advanced reasoning. However, a mismatch between optimal textual and verbal delivery limits their effectiveness in spoken communication. While some approaches adapt LLMs for speech-friendly outputs, their impact on reasoning remains underexplored. We propose **Think-Verbalize-Speak**, a framework that separates reasoning from spoken delivery to preserve the full reasoning capacity of LLMs. Central to our method is verbalizing, an intermediate step that translates thoughts into natural, speech-ready text. We also introduce **ReVerT**, a latency-efficient verbalizer based on incremental and asynchronous summarization.

<p align="center">
    <img src="assets/tvs-framework.png" width="100%"> <br>
</p>

## 🛠️ Installation
- Python >= 3.10
- PyTorch >= 2.5.1
```bash
git clone https://github.com/yhytoto12/TVS-ReVerT.git
cd TVS-ReVerT
conda create -n tvs python=3.10
conda activate tvs
pip install -r requirements.txt

# Use flash attention for faster training and inference (optional)
pip install -U flash-attn --no-build-isolation

# For deepspeed training (optional)
pip install deepspeed
```

## 🤖️ Interactive Demo
You can try our interactive demo by running the following command:

1. Use OpenAI models as a thinking model:
```bash
python demo.py --think_model <openai_model_name> --verbalize_model yhytoto12/revert-Qwen2.5-3B --use_openai_think
```

2. Use Local models as a thinking model:
```bash
# vLLM backend
python -m vllm.entrypoints.transformers --model Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
# In another terminal, run the demo
python demo.py --think_model Qwen/Qwen2.5-7B-Instruct --verbalize_model yhytoto12/revert-Qwen2.5-3B --vllm_url http://localhost:8000/v1
```

## 🚀 Training
### 📊 Datasets

Our training datasets are available on Hugging Face:

- [🤗 **GSM8k**](https://huggingface.co/datasets/yhytoto12/tvs-gsm8k)
- [🤗 **2WikiMultihopQA**](https://huggingface.co/datasets/yhytoto12/tvs-2wikimultihopqa)

These datasets contain thought-verbalization pairs that enable training verbalizers to transform complex reasoning into natural, speech-ready outputs.

### 🛠️ Scripts
You can train the verbalizer using `train/train_*.py` and `scripts/train_*.sh`. The default model is `Qwen/Qwen2.5-3B-Instruct`, but you can change it by modifying the `model_name_or_path` variable in the training scripts.

1. Train **Speech-Friendly Finetuning (SFF)** Model
    ```bash
    bash scripts/train_sff.sh -g <num_gpus>
    ```

2. Train **TVS-SEQ** Model
    ```bash
    bash scripts/train_tvs_seq.sh -g <num_gpus>
    ```

3. Train **TVS-ReVerT** Model
    ```bash
    bash scripts/train_tvs_revert.sh -g <num_gpus>
    ```

    You can find the trained ReVerT models on Hugging Face:
    - [🤗 Qwen2.5-3B ](https://huggingface.co/yhytoto12/revert-Qwen2.5-3B)
    - [🤗 Qwen2.5-0.5B ](https://huggingface.co/yhytoto12/revert-Qwen2.5-0.5B)

## 🖊️ Citation
If you find our ReVerT project useful for your research and applications, please kindly cite using this BibTeX:
```bibtex
@inproceedings{tvs2025@woolee,
  title={Think, Verbalize, then Speak: Bridging Complex Thoughts and Comprehensible Speech},
  author={Sang Hoon Woo, Sehun Lee, Kang-wook Kim, Gunhee Kim},
  booktitle={Proceedings of the EMNLP 2025},
  year={2025}
}
```