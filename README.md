# Sustainable LLM Inference Survey

A survey repository on energy-efficient and sustainable LLM inference techniques.
## What is this Survey About?

This repository provides a comprehensive collection of research papers on sustainable and energy-efficient Large Language Model (LLM) inference. The survey categorizes existing approaches according to different optimization levels and summarizes recent advances in reducing energy consumption, power usage, carbon emissions, latency, and computational cost.

## Why Sustainable LLM Inference?

Large Language Models have achieved remarkable performance across numerous tasks. However, their increasing computational requirements lead to high energy consumption, operational costs, and carbon emissions. Sustainable LLM inference aims to improve efficiency while maintaining model performance through optimizations across different stages of the inference pipeline.

## Table of Contents

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
    - [Batch Tuning](#batch-tuning)
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

- [Hardware-Level Optimization Techniques](#hardware-level-optimization-techniques)
  - [Hardware Power Management](#hardware-power-management)
    - [DVFS / Clock Scaling](#dvfs--clock-scaling)
    - [Power Capping](#power-capping)
  - [Hardware Accelerators](#hardware-accelerators)
    - [Compute-in-Memory (CIM) and Processing-in-Memory (PIM)](#compute-in-memory-cim-and-processing-in-memory-pim)
    - [In-Storage Inference Systems (ISP)](#in-storage-inference-systems-isp)
    - [ASIC / Dataflow Accelerators](#asic--dataflow-accelerators)
    - [TPU / NPU Accelerators](#tpu--npu-accelerators)
    - [FPGA Accelerators](#fpga-accelerators)

- [Cross-Level Optimization Strategies](#cross-level-optimization-strategies)
  - [Input–Deployment Co-optimization](#inputdeployment-co-optimization)
  - [Input–Model Co-optimization](#inputmodel-co-optimization)
  - [Model–Deployment Co-optimization](#modeldeployment-co-optimization)
  - [Deployment–Hardware Co-optimization](#deploymenthardware-co-optimization)
  - [Model–Deployment–Hardware Co-optimization](#modeldeploymenthardware-co-optimization)
  - [Model–Hardware Co-optimization](#modelhardware-co-optimization)

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

  ## Hardware-Level Optimization Techniques

Hardware-level optimization focuses on reducing LLM inference energy by improving hardware operation, resource utilization, and specialized accelerator architectures. Unlike model-level techniques, these approaches optimize the underlying computing platform without modifying the neural network itself.

### Hardware Power Management

#### DVFS / Clock Scaling

- **Energy Efficient GPU Frequency Scaling Policy for Inference Serving Using Queue Model**  
  [[Paper](https://doi.org/10.1109/SCECS65243.2025.11065043)]  
  Technique: Queue-aware DVFS that dynamically adjusts GPU frequency according to workload conditions to improve energy efficiency.

- **Decoupled Analysis of DVFS Effects in Prefill and Decode Stages of Large Language Model Inference**  
  [[Paper](https://doi.org/10.1109/EI268505.2025.11425606)]  
  Technique: Stage-aware DVFS that applies different GPU frequencies during the prefill and decode stages to balance energy consumption and latency.

- **SLO-Aware GPU DVFS for Energy-Efficient LLM Inference Serving**  
  [[Paper](https://doi.org/10.1109/LCA.2024.3406038)]  
  Technique: Dynamically scales GPU frequency while maintaining latency service-level objectives (SLOs).

#### Power Capping

- **From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference**  
  [[Paper](https://doi.org/10.1109/HPEC58863.2023.10363447)]  
  Technique: GPU power capping to reduce inference energy consumption with minimal performance degradation.

- **Sustainable Supercomputing for AI: GPU Power Capping at HPC Scale**  
  [[Paper](https://doi.org/10.1145/3620678.3624793)]  
  Technique: Limits GPU power to reduce energy consumption while maintaining acceptable inference latency.

---

### Hardware Accelerators

#### Compute-in-Memory (CIM) and Processing-in-Memory (PIM)

- **CIM for Transformer Models: Enhancing Large Language Model Inference Efficiency**  
  [[Paper](https://doi.org/10.1109/ISVLSI65124.2025.11130222)]  
  Technique: Compute-in-Memory accelerator that performs computation directly within memory arrays to reduce data movement.

- **A Scalable and Energy-Efficient Processing-in-Memory Architecture for Gen-AI**  
  [[Paper](https://doi.org/10.1109/JETCAS.2025.3566929)]  
  Technique: Processing-in-Memory architecture that executes computation near memory to improve throughput and energy efficiency.

- **PIM-AI: A Novel Architecture for High-Efficiency LLM Inference**  
  [[Paper](https://doi.org/10.48550/arXiv.2411.17309)]  
  Technique: PIM accelerator optimized for transformer and generative AI workloads.

#### In-Storage Inference Systems (ISP)

- **E-Flash: Energy-Efficient LLM Mapping on NAND Flash-Based In-Storage Inference Computing**  
  [[Paper](https://doi.org/10.1109/TVLSI.2026.3657777)]  
  Technique: In-storage processing architecture that performs inference within NAND flash storage to reduce data movement and energy consumption.

- **REIS: A High-Performance and Energy-Efficient Retrieval System with In-Storage Processing**  
  [[Paper](https://doi.org/10.1145/3695053.3731116)]  
  Technique: In-storage retrieval system that integrates retrieval and inference close to storage, improving throughput and energy efficiency.

#### ASIC / Dataflow Accelerators

- **The Immutable Tensor Architecture: A Pure Dataflow Approach for Secure, Energy-Efficient AI Inference**  
  [[Paper](https://doi.org/10.48550/arXiv.2511.22889)]  
  Technique: ASIC/dataflow accelerator that embeds tensor operations in hardware to maximize performance per watt.

#### TPU / NPU Accelerators

- **Leveraging Compute-in-Memory for Efficient Generative Model Inference in TPUs**  
  [[Paper](https://doi.org/10.23919/DATE64628.2025.10993224)]  
  Technique: TPU accelerator using digital compute-in-memory matrix multiplication units for energy-efficient LLM inference.

#### FPGA Accelerators

- **TellMe: An Efficient End-to-End Ternary LLM Prefill and Decode Accelerator with Table-Lookup MatMul on Edge FPGAs**  
  [[Paper](https://doi.org/10.1145/3748173.3779191)]  
  Technique: FPGA accelerator optimized for ternary large language models.

- **HLSTransform: Energy-Efficient LLaMA 2 Inference on FPGAs via High-Level Synthesis**  
  [[Paper](https://doi.org/10.1145/3748173.3779191)]  
  Technique: HLS-based FPGA accelerator that balances throughput, power consumption, and model quality.

- **Fast-Prefill: FPGA Accelerated Sparse Attention for Long Context LLM Prefill**  
  [[Paper](https://doi.org/10.1109/FCCM68464.2026.00067)]  
  Technique: FPGA accelerator for sparse attention to improve long-context LLM inference efficiency.

- **TerEffIC: Highly Efficient Ternary LLM Inference on FPGA**  
  [[Paper](https://doi.org/10.48550/arXiv.2502.16473)]  
  Technique: FPGA accelerator that exploits ternary computation for energy-efficient LLM inference.

  ## Cross-Level Optimization Techniques

Cross-level optimization strategies jointly optimize more than one level of the LLM inference stack. Unlike single-level optimization, these methods coordinate input design, model behavior, deployment decisions, and hardware execution to improve energy efficiency, carbon reduction, latency, and accuracy trade-offs.

### Input–Deployment Co-optimization

- **Prompt Injection Mitigation with Agentic AI, Nested Learning, and AI Sustainability via Semantic Caching**  
  [[Paper](https://doi.org/10.48550/arXiv.2601.13186)]  
  Technique: Combines prompt-level security with semantic caching in a multi-agent deployment framework to reduce redundant LLM calls while improving prompt injection mitigation, latency, energy consumption, and carbon emissions. 

- **Toward Sustainable GenAI Using Generation Directives for Carbon-Friendly Large Language Model Inference**  
  [[Paper](https://doi.org/10.48550/arXiv.2403.12900)]  
  Technique: Uses generation directives at the prompt level together with carbon-aware deployment optimization (Sprout) to reduce the carbon footprint of LLM inference while maintaining generation quality.
  
### Input–Model Co-optimization

- **An Energy-Efficient Vision Language Model Inference with Importance-Aware Token Pruning**  
  [[Paper](https://doi.org/10.1109/APCCAS67402.2025.11376880)]  
  Technique: Combines input-level visual token pruning with model-level importance estimation to remove redundant visual tokens before and during inference, thereby reducing computation and energy consumption while maintaining vision-language task performance.

### Model–Deployment Co-optimization

- **Optimizing Large Language Models: Metrics, Energy Efficiency, and Case Study Insights**  
  [[Paper](https://doi.org/10.1109/CAI64502.2025.00067)]  
  Technique: Jointly evaluates model behavior and deployment-level efficiency metrics to optimize LLM inference energy, latency, and performance.

- **AE-LLM: Adaptive Efficiency Optimization for Large Language Models**  
  [[Paper](https://doi.org/10.48550/arXiv.2603.20492)]  
  Technique: Applies adaptive model-efficiency control during deployment to improve inference efficiency under runtime constraints.

- **E2LLM: Structure-Guided Efficient Inference for LLMs in Distributed Edge-IoT Environments**  
  [[Paper](https://doi.org/10.1109/TMC.2026.3676689)]  
  Technique: Combines model-structure-guided inference with distributed deployment optimization for efficient LLM serving in edge-IoT environments.

### Deployment–Hardware Co-optimization

- **Dynamic Offloading Optimization for Multi-PIM LLM Inference**  
  [[Paper](https://doi.org/10.1109/ICCE-Asia67487.2025.11263662)]  
  Technique: Jointly optimizes deployment offloading decisions and multi-PIM hardware resources for energy-efficient LLM inference.

- **A Deployment-Aware Framework for Carbon- and Water-Efficient LLM Serving**  
  [[Paper](https://doi.org/10.3390/su172310473)]  
  Technique: Combines deployment-aware workload scheduling with infrastructure-level resource management to reduce carbon emissions and water consumption.

- **Characterizing LLM Inference Energy-Performance Tradeoffs Across Workloads and GPU Scaling**  
  [[Paper](https://shashikantilager.com/assets/pdf/publications/LLM_charecterization_ccgrid_2026.pdf)]  
  Technique: Studies deployment workloads together with GPU scaling policies to characterize energy-performance trade-offs.

- **VoltaNaLLM: Feedback-Driven Frequency Control and State-Space Routing for Energy-Efficient LLM Serving**  
  [[Paper](https://doi.org/10.48550/arXiv.2509.04827)]  
  Technique: Combines deployment-level routing with hardware DVFS for energy-efficient LLM serving.

- **GreenLLM: SLO-Aware Dynamic Frequency Scaling for Energy-Efficient LLM Serving**  
  [[Paper](https://doi.org/10.48550/arXiv.2508.16449)]  
  Technique: Integrates deployment-aware serving with hardware frequency scaling while maintaining service-level objectives.

- **FreeSh: Fair, Resource- and Energy-Efficient Scheduling for LLM Serving on Heterogeneous GPUs**  
  [[Paper](https://doi.org/10.48550/arXiv.2508.16449)]  
  Technique: Optimizes deployment scheduling across heterogeneous GPU hardware for fair and energy-efficient inference.

- **WAGES: Workload-Aware GPU Sharing System for Energy-Efficient Serverless LLM Serving**  
  [[Paper](https://doi.org/10.1145/3731599.3767396)]  
  Technique: Combines workload-aware deployment with GPU sharing to improve hardware utilization and reduce energy consumption.

- **A Thermal-Aware Workload Scheduler for High-Performance LLM Inference in Cooling-Regulated Datacenters**  
  [[Paper](https://doi.org/10.1145/3757892.3757906)]  
  Technique: Jointly optimizes deployment scheduling and thermal management of hardware resources.

- **CAMEL: Energy-Aware LLM Inference on Resource-Constrained Devices**  
  [[Paper](https://doi.org/10.48550/arXiv.2508.09173)]  
  Technique: Combines deployment optimization with resource-constrained hardware management for energy-efficient inference.

- **BiScale: Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS**  
  [[Paper](https://doi.org/10.48550/arXiv.2602.18755)]  
  Technique: Integrates disaggregated deployment, phase-aware workload placement, and hardware DVFS.

- **ELLIE: Energy-Efficient LLM Inference at the Edge via Prefill-Decode Splitting**  
  [[Paper](https://doi.org/10.1109/ASAP65064.2025.00031)]  
  Technique: Combines edge deployment with hardware-aware prefill/decode execution for energy-efficient inference.

- **EcoServe: Designing Carbon-Aware AI Inference Systems**  
  [[Paper](https://doi.org/10.48550/arXiv.2502.05043)]  
  Technique: Jointly optimizes deployment strategies and computing infrastructure for carbon-aware AI inference.

- **HYBE: GPU-NPU Hybrid System for Efficient LLM Inference with Million-Token Context Window**  
  [[Paper](https://doi.org/10.1145/3695053.3731051)]  
  Technique: Combines deployment strategies with GPU-NPU heterogeneous hardware execution for efficient long-context inference.

- **Power-Aware Dynamic Reallocation for Inference**  
  [[Paper](https://doi.org/10.48550/arXiv.2601.12241)]  
  Technique: Dynamically reallocates deployment workloads across hardware resources based on power consumption.

- **LEAP: LLM Inference on Scalable PIM-NoC Architecture with Balanced Dataflow and Fine-Grained Parallelism**  
  [[Paper](https://doi.org/10.1109/ICCAD66269.2025.11240722)]  
  Technique: Combines deployment mapping with scalable PIM hardware architecture for energy-efficient inference.

- **AsymServe: Demystifying and Optimizing LLM Serving Efficiency on CPU Acceleration Units**  
  [[Paper](https://link.springer.com/chapter/10.1007/978-981-95-1021-4_17)]  
  Technique: Optimizes LLM serving through deployment-aware scheduling on CPU acceleration hardware.

- **Towards Sustainable Large Language Model Serving**  
  [[Paper](https://doi.org/10.1145/3727200.3727220)]  
  Technique: Jointly optimizes deployment infrastructure and hardware resource management for sustainable LLM serving.

- **Deeploy: Enabling Energy-Efficient Deployment of Small Language Models on Heterogeneous Microcontrollers**  
  [[Paper](https://doi.org/10.1109/TCAD.2024.3443718)]  
  Technique: Integrates deployment optimization with heterogeneous microcontroller hardware for efficient on-device language model inference.

### Model–Deployment–Hardware Co-optimization

- **An Energy-Aware Generative AI Edge Inference Framework for Low-Power IoT Devices**  
  [[Paper](https://www.mdpi.com/2079-9292/14/20/4086)]  
  Technique: Integrates energy-aware model optimization, edge deployment strategies, and low-power hardware management to enable efficient generative AI inference on IoT devices.

- **Optimizing LLM Inference Clusters for Enhanced Performance and Energy Efficiency**  
  [[Paper](https://doi.org/10.36227/techrxiv.172348951.12175366/v1)]  
  Technique: Jointly optimizes model execution, distributed inference deployment, and cluster-level hardware resource allocation to improve performance and energy efficiency.

- **Harnessing Your DRAM and SSD for Sustainable and Accessible LLM Inference with Mixed-Precision and Multi-Level Caching**  
  [[Paper](https://doi.org/10.48550/arXiv.2410.14740)]  
  Technique: Combines mixed-precision model optimization with multi-level caching across DRAM and SSD to improve deployment efficiency and reduce hardware energy consumption.

- **Measurements and Optimizations of the Energy Consumption of AI Systems**  
  [[Paper](https://doi.org/10.36227/techrxiv.175744079.98289886/v1)]  
  Technique: Presents cross-layer optimizations spanning model design, deployment strategies, and hardware execution to reduce the overall energy consumption of AI systems.

- **Disaggregated Speculative Decoding for Carbon-Efficient LLM Serving**  
  [[Paper](https://doi.org/10.1109/LCA.2025.3630094)]  
  Technique: Combines speculative decoding, disaggregated deployment architecture, and hardware resource management to reduce carbon emissions and improve serving efficiency.

- **Energy Considerations of Large Language Model Inference and Efficiency Optimizations**  
  [[Paper](https://aclanthology.org/2025.acl-long.1563.pdf)]  
  Technique: Surveys and evaluates optimization techniques across the model, deployment, and hardware layers, highlighting their combined impact on LLM inference energy efficiency.

### Model–Hardware Co-optimization

