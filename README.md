# Sustainable LLM Inference Survey

A survey repository on energy-efficient and sustainable LLM inference techniques.
## What is this Survey About?

This repository provides a comprehensive collection of research papers on sustainable and energy-efficient Large Language Model (LLM) inference. The survey categorizes existing approaches according to different optimization levels and summarizes recent advances in reducing energy consumption, power usage, carbon emissions, latency, and computational cost.

## Why Sustainable LLM Inference?

Large Language Models have achieved remarkable performance across numerous tasks. However, their increasing computational requirements lead to high energy consumption, operational costs, and carbon emissions. Sustainable LLM inference aims to improve efficiency while maintaining model performance through optimizations across different stages of the inference pipeline.

## Table of Contents

## Table of Contents

- [Input-Level Optimization Techniques](#input-level-optimization-techniques)
  - [Input Length Control](#input-length-control)
    - [Prompt Design](#prompt-design)
    - [Prompt Optimization](#prompt-optimization)
  - [Output Length Control](#output-length-control)
    - [Format Controls](#format-controls)
  - [Stop Criteria](#stop-criteria)
    - [Semantic Stopping](#semantic-stopping)
  - [Context Construction](#context-construction)
    - [Token Pruning](#token-pruning)
    - [RAG Capping](#rag-capping)

- [Model-Level Optimization Techniques](#model-level-optimization-techniques)
  - [Model Size and Architecture Controls](#model-size-and-architecture-controls)
    - [Architecture Choice](#architecture-choice)
    - [Mixture of Attention Spans](#mixture-of-attention-spans)
  - [Numerical Precision and Quantization Controls](#numerical-precision-and-quantization-controls)
    - [Mixed-Precision Inference](#mixed-precision-inference)
    - [Post-Training Quantization](#post-training-quantization)
    - [Activation-Aware Quantization](#activation-aware-quantization)
    - [Hybrid or Mixed-Precision Quantization](#hybrid-or-mixed-precision-quantization)
    - [Adaptive Quantization](#adaptive-quantization)
    - [GGML and GGUF Quantization Formats](#ggml-and-gguf-quantization-formats)
    - [k-Quantization and BitNet Q1.58](#k-quantization-and-bitnet-q158)
  - [Activation Pruning and Freezing](#activation-pruning-and-freezing)
    - [Self-Pruning and Activation Freezing](#self-pruning-and-activation-freezing)
  - [Token and Layer Reduction Controls](#token-and-layer-reduction-controls)
    - [Dynamic Token Compression and Adaptive Layer Pruning](#dynamic-token-compression-and-adaptive-layer-pruning)
  - [Decoding Strategy Controls](#decoding-strategy-controls)
    - [Speculative Decoding](#speculative-decoding)
    - [Speculative Beam Decoding](#speculative-beam-decoding)
    - [Sampling Policy Control](#sampling-policy-control)
      
- [Deployment-Level Optimization Techniques](#deployment-level-optimization-techniques)
  - [Batching and Scheduling](#batching-and-scheduling)
    - [Continuous Batching](#continuous-batching)
    - [Dynamic Batching](#dynamic-batching)
    - [Layered Prefill Scheduling](#layered-prefill-scheduling)
    - [SLIT Scheduling](#slit-scheduling)
  - [Elastic Provisioning Controls](#elastic-provisioning-controls)
    - [Traffic-Aware Autoscaling](#traffic-aware-autoscaling)
    - [Instance Scaling and Idle Instance Shutdown](#instance-scaling-and-idle-instance-shutdown)
  - [Memory Management](#memory-management)
    - [KV-Cache Reuse and Prefix Caching](#kv-cache-reuse-and-prefix-caching)
    - [Pre-Stored Attention Matrices](#pre-stored-attention-matrices)
  - [Semantic Caching](#semantic-caching)
    - [Semantic Response Caching](#semantic-response-caching)
  - [Routing and Model Selection Controls](#routing-and-model-selection-controls)
    - [Task-Aware Model Selection](#task-aware-model-selection)
    - [Confidence-Based Routing](#confidence-based-routing)
    - [Dynamic Routing](#dynamic-routing)
    - [Context Routing](#context-routing)
    - [Energy-Aware Routing](#energy-aware-routing)
    - [Complexity-Aware Query Routing](#complexity-aware-query-routing)
    - [Environmental-Cost Routing](#environmental-cost-routing)
    - [Carbon-Aware Scheduling and Workload Allocation](#carbon-aware-scheduling-and-workload-allocation)
  - [Hardware-Aware Scheduling and Resource Allocation](#hardware-aware-scheduling-and-resource-allocation)
    - [Heterogeneous GPU Allocation](#heterogeneous-gpu-allocation)
    - [GPU Scaling](#gpu-scaling)
    - [CPU/GPU Resource Allocation](#cpugpu-resource-allocation)
    - [Compiler Optimization](#compiler-optimization)
    - [Fleet-Level Optimization](#fleet-level-optimization)
  - [Parallelism and Disaggregated Serving](#parallelism-and-disaggregated-serving)
    - [Right-Sized Parallelism](#right-sized-parallelism)
    - [Disaggregated Serving](#disaggregated-serving)
  - [Other Deployment-Level Efficiency Techniques](#other-deployment-level-efficiency-techniques)
    - [Batch Tuning](#batch-tuning)
    - [Cost-Aware Scheduling](#cost-aware-scheduling)

## Input-Level Optimization Techniques

### Input Length Control

#### Prompt Design

- Green Prompt Engineering for Sustainable Generative AI, *Empirical Software Engineering, 2026*. [[Paper](https://doi.org/10.1016/j.ese.2026.100684)]

#### Prompt Optimization

- Energy-Aware Prompt Optimization for Large Language Models: Balancing Accuracy, Cost, and Sustainability, *TechRxiv, 2026*. [[Paper](https://doi.org/10.36227/techrxiv.175615629.90190202/v1)]

### Output Length Control

#### Format Controls

- Are LLMs Ready for TOON? Benchmarking Structural Correctness–Sustainability Trade-offs in Novel Structured Output Formats, *arXiv, 2026*. [[Paper](https://doi.org/10.48550/arXiv.2601.12014)]

### Stop Criteria

#### Semantic Stopping

- Towards Green AI: Decoding the Energy of LLM Inference in Software Development, *arXiv, 2026*. [[Paper](https://arxiv.org/abs/2602.05712)]

### Context Construction

#### Token Pruning

- Adaptive Neural Token Streaming: A Novel Approach for Optimizing Large Language Model Efficiency and Accuracy, *TechRxiv, 2025*. [[Paper](https://doi.org/10.36227/techrxiv.173014495.51612297/v1)]
  

#### RAG Capping

- On the Effectiveness of Proposed Techniques to Reduce Energy Consumption in RAG Systems: A Controlled Experiment, *ACM, 2026*. [[Paper](https://doi.org/10.1145/3786581.3786932)]

## Model-Level Optimization Techniques

Model-level optimization focuses on improving the internal structure, numerical representation, activation behavior, token processing, and decoding strategy of LLMs to reduce energy, carbon, power, latency, and memory cost.

### Model Size and Architecture Controls

#### Architecture Choice

- **TokenPowerBench: Benchmarking the Power Consumption of LLM Inference**  
  [[Paper](https://arxiv.org/abs/2512.03024)]  
  Technique: Architecture choice using dense/MoE and layer/width variants.

#### Mixture of Attention Spans

- **Mixture of Attention Spans: Optimizing LLM Inference Efficiency with Heterogeneous Sliding-Window Lengths**  
  [[Paper](https://doi.org/10.48550/arXiv.2406.14909)]  
  Technique: Assigns different sliding-window attention spans to attention heads and transformer layers.

### Numerical Precision and Quantization Controls

#### Mixed-Precision Inference

- **Understanding Efficiency: Quantization, Batching, and Serving Strategies in LLM Energy Use**  
  [[Paper](https://arxiv.org/abs/2601.22362)]  
  Technique: Uses lower-precision formats such as FP16/BF16 to reduce inference energy.

#### Post-Training Quantization

- **Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency**  
  [[Paper](https://arxiv.org/abs/2504.03360)]  
  Technique: Applies post-training quantization to reduce inference energy.

#### Activation-Aware Quantization

- **From Prompts to Power: Measuring the Energy Footprint of LLM Inference**  
  [[Paper](https://arxiv.org/abs/2511.05597)]  
  Technique: Uses Activation-Aware Weight Quantization (AWQ) to preserve important weights while reducing precision.

#### Hybrid or Mixed-Precision Quantization

- **Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective**  
  [[Paper](https://doi.org/10.48550/arXiv.2508.16712)]  
  Technique: Hybrid quantization using different precision levels across model components.

- **FGMP: Fine-Grained Mixed-Precision Weight and Activation Quantization for Hardware-Accelerated LLM Inference**  
  [[Paper](https://doi.org/10.48550/arXiv.2504.14152)]  
  Technique: Fine-grained mixed-precision quantization that preserves sensitive blocks in higher precision.

#### Adaptive Quantization

- **Green Transformers: Leveraging Neuromorphic Computing, Sparse Attention, and Adaptive Quantization for Energy-Efficient Inference**  
  [[Paper](https://doi.org/10.36227/techrxiv.176127449.98578610/v1)]  
  Technique: Dynamically adjusts numerical precision according to input complexity and energy metrics.

#### GGML and GGUF Quantization Formats

- **Benchmarking Emerging Deep Learning Quantization Methods for Energy Efficiency**  
  [[Paper](https://doi.org/10.1109/ICSA-C63560.2024.00049)]  
  Technique: Evaluates GGML and GGUF lightweight quantized model formats for efficient inference.

#### k-Quantization and BitNet Q1.58

- **LLMPI: Optimizing LLMs for High-Throughput on Raspberry Pi**  
  [[Paper](https://openaccess.thecvf.com)]  
  Technique: Uses k-Quantization and BitNet Q1.58 to improve energy efficiency on resource-constrained devices.

### Activation Pruning and Freezing

#### Self-Pruning and Activation Freezing

- **Eco-Transformers: Carbon-Efficient and Accelerated Inference through Self-Pruning and Activation Freezing in Large Language Models**  
  [[Paper](https://doi.org/10.1109/ICNGCS64900.2025.11183217)]  
  Technique: Uses dynamic weight masking and activation freezing to reduce computation and carbon emissions.

### Token and Layer Reduction Controls

#### Dynamic Token Compression and Adaptive Layer Pruning

- **Improve the Accuracy and Efficiency of Large Language Models via Dynamic Token Compression and Adaptive Layer Pruning**  
  [[Paper](https://doi.org/10.36227/techrxiv.173030653.35636983/v1)]  
  Technique: Compresses redundant tokens and skips unnecessary transformer layers based on input complexity.

### Decoding Strategy Controls

#### Speculative Decoding

- **Multitoken Joint Speculative Decoding for Accelerating Large Language Model Inference**  
  [[Paper](https://arxiv.org/abs/2407.09722)]  
  Technique: Uses a draft-and-verify strategy to accelerate autoregressive decoding.

#### Speculative Beam Decoding

- **Dynamic-Width Speculative Beam Decoding for LLM Inference**  
  [[Paper](https://doi.org/10.1609/aaai.v39i23.34690)]  
  Technique: Combines speculative decoding with beam search to improve inference efficiency.

#### Sampling Policy Control

- **Impact of Decoding Strategies on GPU Energy Usage in Large Language Model Text Generation**  
  [[Paper](https://doi.org/10.1038/s41598-025-31896-0)]  
  Technique: Evaluates top-k, top-p, beam search, and temperature-based sampling strategies for energy-efficient generation.

  ## Deployment-Level Optimization Techniques

Deployment-level optimization focuses on improving how LLM inference requests are served, scheduled, routed, cached, and allocated across computing resources to reduce energy consumption, carbon emissions, power usage, latency, and serving cost.

### Batching and Scheduling

#### Continuous Batching

- **Energy Efficiency in Large Language Models: An Empirical Study**  
  [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-92734-8_22)]  
  Technique: Continuous batching.

#### Dynamic Batching

- **Inference Acceleration for Large Language Models on CPUs**  
  [[Paper](https://www.preprints.org/manuscript/202402.1702)]  
  Technique: Dynamic batching.

#### Batch Tuning

- **Measuring and Improving the Energy Efficiency of Large Language Models Inference**  
  [[Paper](https://doi.org/10.1109/ACCESS.2024.3409745)]  
  Technique: Batch tuning.

#### Layered Prefill Scheduling

- **From Tokens to Layers: Redefining Stall-Free Scheduling for LLM Serving with Layered Prefill**  
  [[Paper](https://doi.org/10.48550/arXiv.2510.08055)]  
  Technique: Layered prefill scheduling.

#### SLIT Scheduling

- **Sustainable Carbon-Aware and Water-Efficient LLM Scheduling in Geo-Distributed Cloud Datacenters**  
  [[Paper](https://doi.org/10.1145/3716368.3735301)]  
  Technique: SLIT scheduling.

---

### Elastic Provisioning Controls

#### Traffic-Aware Autoscaling

- **throttLL'eM: Predictive GPU Throttling for Energy Efficient LLM Inference Serving**  
  [[Paper](https://doi.org/10.1109/HPCA61900.2025.00103)]  
  Technique: Traffic-aware autoscaling.

#### Instance Scaling and Idle Instance Shutdown

- **DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency**  
  [[Paper](https://doi.org/10.1109/HPCA61900.2025.00102)]  
  Technique: Instance scaling and idle instance shutdown.

---

### Memory Management

#### KV-Cache Reuse and Prefix Caching

- **Cache Your Prompt When It's Green—Carbon-Aware Caching for Large Language Model Serving**  
  [[Paper](https://doi.org/10.1145/3788087)]  
  Technique: KV-cache reuse and prefix caching.

#### Pre-Stored Attention Matrices

- **StoreLLM: Energy Efficient Large Language Model Inference with Permanently Pre-Stored Attention Matrices**  
  [[Paper](https://doi.org/10.1145/3679240.3734604)]  
  Technique: Pre-stored attention matrices.

---

### Semantic Caching

#### Semantic Response Caching

- **Cache Saver: A Modular Framework for Efficient, Affordable, and Reproducible LLM Inference**  
  [[Paper](https://aclanthology.org/anthology-files/pdf/findings/2025.findings-emnlp.1402.pdf)]  
  Technique: Semantic response caching.

---

### Routing and Model Selection Controls

#### Task-Aware Model Selection

- **GreenServ: Energy-Efficient Context-Aware Dynamic Routing for Multi-Model LLM Inference**  
  [[Paper](https://arxiv.org/abs/2601.17551)]  
  Technique: Task-aware model selection.

#### Confidence-Based Routing

- **HybridServe: Efficient Serving of Large AI Models with Confidence-Based Cascade Routing**  
  [[Paper](https://doi.org/10.1109/ICDCSW63273.2025.00028)]  
  Technique: Confidence-based routing.

- **Reducing Environmental Costs Whilst Maintaining Operational Effectiveness in Large Language Models Through Chain Routing**  
  [[Paper](https://aclanthology.org/2025.konvens-2.20.pdf)]  
  Technique: Environmental-cost routing.

#### Dynamic Routing

- **Smart Meta-Controller: Energy Efficient Dynamic Routing for Large Language Model Inference Systems**  
  [[Paper](https://doi.org/10.1109/CSITSS67709.2025.11295492)]  
  Technique: Dynamic routing.

#### Context Routing

- **Sustainable LLM Inference Using Context-Aware Model Switching**  
  [[Paper](https://arxiv.org/abs/2602.22261)]  
  Technique: Context routing.

#### Energy-Aware Routing

- **PEARL: Performance and Energy Aware Routing for LLMs**  
  [[Paper](https://doi.org/10.1016/j.future.2025.108218)]  
  Technique: Energy-aware routing.

#### Complexity-Aware Query Routing

- **EcoThink: A Green Adaptive Inference Framework for Sustainable and Accessible Agents**  
  [[Paper](https://doi.org/10.1145/3774904.3792995)]  
  Technique: Complexity-aware query routing.

#### Carbon-Aware Scheduling and Workload Allocation

- **GAR: Carbon-Aware Routing for LLM Inference via Constrained Optimization**  
  [[Paper](https://doi.org/10.48550/arXiv.2605.11603)]  
  Technique: Carbon-aware routing.

- **Carbon-Aware Workload Shifting for Mitigating Environmental Impact of Generative AI Models**  
  [[Paper](https://doi.org/10.1109/iThings-GreenCom-CPSCom-SmartData-Cybermatics62450.2024.00087)]  
  Technique: Carbon-aware workload allocation.

- **Carbon-Aware Quality Adaptation for Energy-Intensive Services**  
  [[Paper](https://doi.org/10.1145/3679240.3734614)]  
  Technique: Carbon-aware scheduling.

---

### Hardware-Aware Scheduling and Resource Allocation

#### Heterogeneous GPU Allocation

- **Idle Consumer GPUs as a Complement to Enterprise Hardware for LLM Inference: Performance, Cost and Carbon Analysis**  
  [[Paper](https://doi.org/10.1145/3775043.3775047)]  
  Technique: Heterogeneous GPU allocation.

#### GPU Scaling

- **Towards Greener LLMs: Bringing Energy-Efficiency to the Forefront of LLM Inference**  
  [[Paper](https://arxiv.org/abs/2403.20306)]  
  Technique: GPU scaling.

#### CPU/GPU Resource Allocation

- **Evaluating the Carbon Impact of Large Language Models at the Inference Stage**  
  [[Paper](https://doi.org/10.1109/IPCCC59175.2023.10253886)]  
  Technique: CPU/GPU resource allocation.

#### Compiler Optimization

- **Energy-Efficient Large Language Models**  
  [[Paper](https://doi.org/10.1016/j.future.2026.108483)]  
  Technique: Compiler optimization.

#### Fleet-Level Optimization

- **The 1/W Law: An Analytical Study of Context-Length Routing Topology and GPU Generation Gains for LLM Inference Energy Efficiency**  
  [[Paper](https://doi.org/10.48550/arXiv.2603.17280)]  
  Technique: Fleet-level optimization.

---

### Parallelism and Disaggregated Serving

#### Right-Sized Parallelism

- **DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency**  
  [[Paper](https://doi.org/10.1109/HPCA61900.2025.00102)]  
  Technique: Right-sized parallelism.

#### Disaggregated Serving

- **Splitwise: Efficient Generative LLM Inference Using Phase Splitting**  
  [[Paper](https://doi.org/10.1109/ISCA59077.2024.00019)]  
  Technique: Disaggregated serving.
