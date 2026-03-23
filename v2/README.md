## Repository Additions for Baseline Plotting and Compute Skipping

The sections below summarize the code added or modified for the baseline plotting and compute skipping experiments. The original project README starts immediately after this overview.

```text
v2/
├── eval.py
├── generation_functions.py
├── baseline-plotting/
│   ├── baseline_trace.py
│   ├── modeling.py
│   ├── part-a-scripts/
│   ├── part-b-scripts/
│   └── part-c-scripts/
└── compute-skipping/
    ├── token-level/
    │   ├── token_skip_config.py
    │   ├── token_skip_policy.py
    │   ├── token_skip_manager.py
    │   ├── token_skip_stats.py
    │   └── modeling.py
    ├── layer-level/
    │   ├── layer_skip_config.py
    │   ├── layer_skip_policy.py
    │   ├── layer_skip_manager.py
    │   ├── layer_skip_stats.py
    │   └── modeling.py
    └── plotting/
        └── plot_accuracy_vs_flops.py
```

### Top-Level Runtime Files

- `eval.py`: Main evaluation entry point that loads the model, parses experiment settings, and launches `lm_eval` runs. I added the token-level and layer-level skip configuration hooks here so each setting can attach the correct manager and stats recorder during evaluation.
- `generation_functions.py`: Core decode-time sampling logic for Fast-dLLM generation. I added the per-sample recorder setup, denoising-step tracking, and skip-context plumbing so the compute-skipping experiments can log step counts and layer-step statistics.

### `baseline-plotting/`

- `baseline-plotting/baseline_trace.py`: Recorder used for the baseline analysis experiments. It saves per-sample hidden-state, attention, and FFN trace data into JSON files that the plotting scripts read later.
- `baseline-plotting/modeling.py`: Baseline-trace-only model variant with the tracing hooks added around the forward pass. This file was meant to be copy-pasted into the active runtime `modeling.py` when collecting baseline trace logs, and it isolates only the baseline plotting modifications.
- `baseline-plotting/part-a-scripts/`: Plotting scripts for Part A baseline analysis. These scripts generate the layer-level hidden-state heatmaps and 100-sample summary plots from the baseline trace logs.
- `baseline-plotting/part-b-scripts/`: Plotting scripts for Part B baseline analysis. These scripts generate the token-level hidden-state plots and the hidden-vs-attention / hidden-vs-FFN scatter plots from the baseline trace logs.
- `baseline-plotting/part-c-scripts/`: Plotting scripts for Part C baseline analysis. These scripts generate the attention similarity, attention range, and FFN similarity figures from the baseline trace logs.

### `compute-skipping/layer-level/`

- `compute-skipping/layer-level/layer_skip_config.py`: Configuration object for the layer-level skipping experiments. It stores the enabled state, aggregation mode, threshold, and naming logic for each layer-skip setting.
- `compute-skipping/layer-level/layer_skip_policy.py`: Policy logic for deciding whether an entire layer should be reused. It computes per-token cosine similarity across adjacent denoising steps and reduces those scores with `avg` or `max` before applying the threshold.
- `compute-skipping/layer-level/layer_skip_manager.py`: Runtime state manager for layer-level skipping. It caches previous-step layer inputs and outputs, builds the per-layer reuse plan, and writes the skip statistics needed for later FLOPs accounting.
- `compute-skipping/layer-level/layer_skip_stats.py`: Logging utility for layer-level experiments. It saves per-sample denoising-step counts and per-layer skip records in the JSON format used by the analysis scripts.
- `compute-skipping/layer-level/modeling.py`: Layer-skip-only model variant with the layer-level reuse logic added to the decoder stack. This file was meant to be copy-pasted into the active runtime `modeling.py` when running layer-level skipping experiments, and it isolates only the layer-level skipping modifications.

### `compute-skipping/token-level/`

- `compute-skipping/token-level/token_skip_config.py`: Configuration object for the token-level skipping experiments. It stores the enabled state, threshold or top-k settings, and naming logic for each token-skip run.
- `compute-skipping/token-level/token_skip_policy.py`: Policy logic for deciding which tokens stay active in a layer. It computes per-token cosine similarity across adjacent denoising steps and builds either threshold-based or top-k active/reuse masks.
- `compute-skipping/token-level/token_skip_manager.py`: Runtime state manager for token-level skipping. It caches previous-step layer inputs and outputs, builds the active-token reuse plan, and logs the per-layer token counts used in the FLOPs calculation.
- `compute-skipping/token-level/token_skip_stats.py`: Logging utility for token-level experiments. It saves per-sample denoising-step counts plus per-layer token activity records in the JSON format used by the plotting scripts.
- `compute-skipping/token-level/modeling.py`: Token-skip-only model variant with the active-token attention and MLP logic added to the decoder stack. This file was meant to be copy-pasted into the active runtime `modeling.py` when running token-level skipping experiments, and it isolates only the token-level skipping modifications.

### `compute-skipping/plotting/`

- `compute-skipping/plotting/plot_accuracy_vs_flops.py`: Final summary plotting script for the compute-skipping experiments. It reads the logged skip statistics and resolved accuracies, computes theoretical FLOPs reduction, and generates the final Accuracy vs FLOPs plot plus the average denoising-steps table.

# Fast-dLLM v2: Efficient Block-Diffusion Large Language Model

[![Project](https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages)](https://nvlabs.github.io/Fast-dLLM/v2)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2509.26328)
[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B)

Fast-dLLM v2 is a carefully designed block diffusion language model (dLLM) that efficiently adapts pretrained autoregressive (AR) models into dLLMs for parallel text generation, requiring only approximately 1B tokens of fine-tuning. This represents a **500x reduction** in training data compared to full-attention diffusion LLMs while preserving the original model's performance.

## 🎬 Demo
https://github.com/user-attachments/assets/f2e055f5-3a44-41ca-9ef8-c84cf3ac2951

## 🎯 Key Features

### 1. **Block Diffusion Mechanism**
- Novel training recipe combining block diffusion with complementary attention masks
- Enables blockwise bidirectional context modeling 
- Token shift mechanism to retain autoregressive characteristics

<div align="center">
  <img src="asset/training_recipe.png" alt="Training Recipe" width="700"/>
  <p><em>Block-wise causal attention mask and complementary training strategy</em></p>
</div>

### 2. **Hierarchical Caching System**
- **Block-level cache**: Stores historical context representations across blocks
- **Sub-block cache**: Enables efficient parallel generation within partially decoded blocks

### 3. **Parallel Decoding Pipeline**
- Achieves up to **2.5x speedup** over standard AR decoding
- Real-time visualization of the denoising process
- Maintains generation quality while delivering state-of-the-art efficiency

<div align="center">
  <img src="asset/visualization_animation.gif" alt="Generation Process Visualization" width="700"/>
  <p><em>Block-level autoregressive generation with sub-block parallelization</em></p>
</div>

## 🚀 Performance

### Throughput Comparison
Fast-dLLM v2 significantly outperforms baselines in both efficiency and accuracy:
- **2.54× higher throughput** than Qwen2.5-7B-Instruct
- **5.2% accuracy improvement** over Fast-dLLM-LLaDA

<div align="center">
  <img src="asset/throughput.png" alt="Throughput Comparison" width="700"/>
  <p><em>Throughput and accuracy comparison across different model variants</em></p>
</div>

### Benchmark Results
Comprehensive evaluation across diverse tasks:

| Model Size | Model | HumanEval-Base | HumanEval-Plus| MBPP-Base | MBPP-Plus |  GSM8K | Math | IFEval | MMLU | GPQA | Average |
|------------|-------|-----------|--|---|---|-------|------|--------|------|------|---------|
| **1B-scale** | Fast-dLLM v2 (1.5B) | 43.9 | 40.2 | 50.0 | 41.3 | 62.0 | 38.1  | 47.0 | 55.1 | 27.7 | **45.0** |
| **7B+ scale** | Fast-dLLM v2 (7B) | 63.4 | 58.5 | 63.0 | 52.3 | 83.7 | 61.6 | 61.4 | 66.6 | 31.9 | **60.3** |

<div align="center">
  <img src="asset/benchmark_results.png" alt="Benchmark Results" width="800"/>
  <p><em>Comprehensive benchmark comparison across diverse tasks</em></p>
</div>


## 🏋️ Training

### Environment Setup
First, create and activate a conda environment:

```bash
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
```

### Installation
Install the package in development mode:

```bash
pip install -e .
```

### Data Preparation
Download the training data (e.g., Alpaca dataset):

```bash
cd data
bash download.sh alpaca
```

### Fine-tuning
Run the fine-tuning script:

```bash
bash train_scripts/finetune_alpaca.sh
```

This will start the training process using the Alpaca dataset with the optimized block diffusion training recipe.

## 🎮 Quick Start

### Interactive Chatbot
Launch the Gradio-based web interface:

```bash
python app.py
```

This will start a web server at `http://localhost:10086` with:
- Real-time conversation interface
- Live visualization of the denoising process
- Adjustable generation parameters (block size, temperature, threshold)
- Performance metrics display

### Command Line Chat
For a simple command-line interface:

```bash
python run_chatbot.py
```

Commands:
- Type your message and press Enter
- `clear` - Clear conversation history
- `exit` - Quit the chatbot


## 📊 Evaluation

### Run Benchmark Evaluation
Execute the evaluation script for comprehensive benchmarking:

```bash
bash eval_script.sh
```

This script evaluates the model on:
- **MMLU**: Massive Multitask Language Understanding
- **GPQA**: Graduate-level Google-Proof Q&A
- **GSM8K**: Grade School Math 8K
- **Minerva Math**: Mathematical reasoning
- **IFEval**: Instruction following evaluation

### Custom Evaluation
For custom evaluation with specific parameters:

```bash
accelerate launch eval.py \
    --tasks gsm8k \
    --batch_size 32 \
    --num_fewshot 0 \
    --model fast_dllm_v2 \
    --model_args model_path=Efficient-Large-Model/Fast_dLLM_v2_7B,threshold=0.9
```

## 🏗️ Architecture

### Training Recipe
- **Token Shift Mechanism**: Each masked token is predicted using the logit of its preceding token
- **Block-wise Causal Attention**: Access to all clean tokens from previous blocks and noisy tokens within current block
- **Complementary Masks**: Alternate masking patterns ensure every token position is learned

### Generation Process
1. **Block-level Generation**: Autoregressive at the block level
2. **Sub-block Parallelization**: Parallel decoding within blocks for efficiency
3. **Hierarchical Caching**: Block and sub-block level caching for speed optimization

## 📁 File Structure

```
v2/
├── app.py                    # Gradio web interface
├── run_chatbot.py           # Command-line chatbot
├── eval.py                  # Evaluation harness integration
├── eval_script.sh           # Benchmark evaluation script
├── generation_functions.py  # Core generation algorithms
├── index.html              # Project webpage
├── asset/                  # Visual assets
│   ├── demo.mp4
│   ├── benchmark_results.png
│   ├── throughput.png
│   ├── training_recipe.png
│   └── visualization_animation.gif
└── README.md               # This file
```

## 🎨 Visualization Features

The web interface provides real-time visualization of:
- **Denoising Process**: Watch tokens being unmasked in real-time
- **Generation Progress**: Visual feedback of the generation pipeline
- **Performance Metrics**: Live throughput and timing information
- **Slow Motion Replay**: Detailed step-by-step visualization

## 🔬 Technical Details

### Model Architecture
- Based on Qwen2.5 architecture with block diffusion modifications
- 7B parameter model with efficient parallel decoding capabilities
- Custom attention mechanisms for block-wise processing

### Optimization Techniques
- Block-level KV caching for reduced computation
- Sub-block parallel processing for improved throughput
- Confidence-aware token unmasking for quality preservation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](../CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](../LICENSE) file for details.

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{wu2025fastdllmv2efficientblockdiffusion,
      title={Fast-dLLM v2: Efficient Block-Diffusion LLM}, 
      author={Chengyue Wu and Hao Zhang and Shuchen Xue and Shizhe Diao and Yonggan Fu and Zhijian Liu and Pavlo Molchanov and Ping Luo and Song Han and Enze Xie},
      year={2025},
      eprint={2509.26328},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.26328}, 
}
```

## 🙏 Acknowledgements

We thank [Qwen2.5](https://github.com/QwenLM/Qwen2.5) for the base model architecture
