# A Comprehensive Survey on Efficient Inference and Small Language Models: Architectural Innovations, Quantization, KV Cache Optimization, and Speculative Decoding

## TL;DR
- Small Language Models (SLMs) and efficient inference frameworks mitigate memory bandwidth bottlenecks and compute latency through advanced quantization, KV cache management, and speculative decoding [1][2].
- Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT), alongside extreme 1.58-bit ternary models, replace costly floating-point multiplications with integer operations to dramatically shrink memory footprint [1][3].
- System-aware KV cache mechanisms—including tensor virtualization, tiered offloading, and structural pruning/compression—overcome long-context memory overheads, though aggressive token eviction risks instruction-following degradation [4][5].
- Speculative decoding paradigms and parallel multi-branch draft architectures (such as Medusa and EAGLE-2) accelerate auto-regressive generation speed by 2x to 5x while preserving output distribution fidelity [6][7].

## Background
Auto-regressive large and small language models generate text sequentially, token by token. This iterative generation model creates a severe memory-bandwidth bottleneck: during every token generation step, the entire model parameter weight matrix must be fetched from High-Bandwidth Memory (HBM) or DRAM into processor SRAM. As model scale increases or sequence length expands, memory transfer bandwidth saturation limits token generation throughput. 

Concurrently, the Key-Value (KV) cache stores intermediate attention tensors across sequence steps to avoid recomputing past token representations. However, the KV cache grows linearly with sequence length and batch size, eventually exhausting GPU memory in long-context workloads [4][8]. 

To address these hardware and memory challenges, efficient inference research intersects four main pillars:
1. **Quantization and Low-Bit Representations**: Compressing parameter weights and activations from FP16/BF16 down to INT8, INT4, INT2, or ternary ({-1, 0, 1}) values to reduce transfer volume and accelerate arithmetic [1][9][3].
2. **KV Cache Optimization**: Managing memory allocation, multi-tier offloading (CPU/GPU), and token compression or eviction to sustain long-context inference without exceeding memory limits [4][8][5].
3. **Speculative Decoding and Draft Verification**: Leveraging lightweight draft models or multi-head architectures to generate candidate token sequences in parallel, verifying them in a single target-model forward pass [6][7].
4. **SLM Architecture Design and Distillation**: Designing compact transformer architectures optimized for edge deployment, backed by over-training, structured pruning, and knowledge distillation [2][10].

Historically, scaling laws dictated ever-larger models, but recent findings demonstrate that "over-trained" Small Language Models (SLMs) running on optimized inference runtimes can match or exceed larger legacy systems while drastically reducing serving costs [2][11].

## Quantization and Quantization-Aware Training (QAT)
Quantization maps high-precision floating-point parameters to discrete lower-precision representations, directly addressing the memory-bandwidth bottleneck of auto-regressive decoding. Post-Training Quantization (PTQ) methods approximate weights without full retraining. For instance, GPTQ utilizes second-order Hessian information based on the Optimal Brain Surgeon framework to compensate for weight rounding errors, providing robust multi-layer accuracy preservation [1]. AWQ (Activation-Aware Weight Quantization) protects a small fraction of salient weights associated with large input activations in high precision, avoiding complex second-order Hessian inversions [1]. To address outlier accumulation, recent advancements like SpQR isolate outlier weights in higher precision while storing bulk weights in extreme low bit-widths, and orthogonal rotation methods (such as SpinQuant and KronQ) reduce weight-activation outlier correlations [1].

Beyond conventional INT4/INT8 quantization, extreme low-bit paradigms have emerged. BitNet b1.58 introduces a ternary representation where every parameter takes values in $\{-1, 0, 1\}$, consuming 1.58 bits per parameter [3]. By substituting standard linear layers with `BitLinear`, floating-point multiplication operations are eliminated entirely, replaced by highly efficient integer addition and subtraction [3]. Similarly, ParetoQ establishes a unified scaling framework spanning 1-bit through 4-bit quantization, demonstrating that sub-4-bit regimes maintain highly competitive size-accuracy tradeoffs when paired with optimized scaling calibration [9].

Quantization-Aware Training (QAT), through methods like LLM-QAT and EfficientQAT, simulates quantization noise during fine-tuning using straight-through estimators (STE) or generative distillation [1]. QAT consistently outperforms PTQ at extreme sub-2-bit and 2-bit levels by allowing network parameters to adapt directly to discrete numerical constraints. These algorithmic breakthroughs are supported by production serving frameworks (such as vLLM, TensorRT-LLM, and LMDeploy) that integrate optimized mixed-precision kernels to accelerate token generation [1].

## KV Cache Compression, Eviction, and Optimization Mechanisms
As context windows expand to tens or hundreds of thousands of tokens, the Key-Value (KV) cache dominates memory consumption and restricts batch sizes. System-aware KV cache optimization taxonomies classify solutions into temporal (scheduling), spatial (placement and migration), and structural (representation and retention) dimensions [4][8]. 

At the runtime level, **PagedAttention** (popularized by vLLM) virtualizes KV memory into fixed-size physical blocks managed via page tables, eliminating internal and external memory fragmentation and enabling efficient prefix sharing [8]. Simultaneously, IO-aware attention kernels like **FlashAttention** utilize tiled memory access and online softmax to avoid materializing the massive $N \times N$ attention matrix in high-bandwidth memory.

To control long-context memory growth, structural compression and token eviction strategies are deployed. StreamingLLM retains attention sinks and rolling recent windows, while H2O (Heavy-Hitter Oracle) and SnapKV dynamically identify and retain critical attention tokens [4][5]. For multi-tier memory hierarchies, frameworks like **ShadowKV** offload value caches to CPU memory while keeping low-rank pre-RoPE key caches on GPU, overlapping cache transfers via CUDA multi-streams to achieve up to $3\times$ higher throughput [5]. Similarly, SpeCache prefetches KV pairs based on estimated importance across tiered CPU-GPU storage [12]. 

However, recent empirical analyses highlight important trade-offs: aggressive token eviction can introduce instruction-following degradation, multi-instruction blind spots, and system prompt leakage if critical instructional tokens are inadvertently discarded [4].

## Speculative Decoding and Draft-Verify Architectures
Speculative decoding accelerates auto-regressive generation without sacrificing output distribution fidelity by pairing a fast, lightweight draft model with a larger, slower target model [6][7]. The draft model generates a sequence of candidate tokens autoregressively, and the target model evaluates all candidates in a single parallel forward pass.

Traditional speculative pipelines relied on fixed sequential draft structures, but recent advances focus on dynamic, multi-branch architectures and training objectives. **Medusa** eliminates external draft models entirely by augmenting the backbone LLM with multiple parallel decoding heads to predict subsequent tokens simultaneously, achieving 2.2x to 2.8x speedups [6]. **EAGLE-2** builds context-aware dynamic draft trees on-the-fly using well-calibrated draft model confidence scores as acceptance proxies, achieving acceleration ratios exceeding 4x [7].

Algorithmic refinements have also improved verification and training efficiency. Training objectives that directly minimize expected decoding rounds resolve the mismatch of block-local surrogates [13]. Nucleus Speculative Decoding (NSD) introduces plausibility-aware verification beyond strict exact-distribution correction to retain highly plausible draft tokens [14]. Furthermore, dynamic controllers like APEX adapt drafting depth based on context predictability [15], while DLoop eliminates redundant verification forward passes when draft tokens are fully accepted [16].

## SLM Architecture Design, Distillation, and Scaling Laws
Small Language Models (SLMs) (typically ranging from 100M to 5B parameters) are engineered for resource-constrained edge devices and high-throughput server deployments [2][17]. Architectural designs prioritize memory efficiency through Grouped-Query Attention (GQA), RMSNorm, SwiGLU activations, and shared embedding/lm_head weights (e.g., in Gemma and Qwen) [2][11]. Moreover, compact vocabulary sizing (e.g., 32k tokens) is crucial to minimize the compute buffer memory footprint ($batch\_size \times seq\_len \times vocab\_size$), which otherwise consumes disproportionate memory on edge hardware [17].

To match the reasoning and generalization capabilities of larger models, SLMs are heavily "over-trained" relative to traditional Chinchilla scaling laws, frequently ingesting 1.5T to 12T+ high-quality tokens [2][11]. High-quality synthetic and model-filtered pre-training datasets (such as FineWeb-Edu and DCLM) bridge the capability gap between open-source SLMs and proprietary models [2]. When training from scratch is impractical, structured compression techniques—such as ShortGPT's Block Influence pruning and Generalized Knowledge Distillation (GKD) with skew KL divergence—effectively distill large teacher models into high-performance compact SLMs [10][18].

## Trends and Open Problems
The convergence of efficient inference algorithms and small language models has unlocked rapid edge deployment and massive server-side cost reductions, yet several critical open problems remain:
- **Quantization Degradation at Sub-1-Bit Scales:** While 1.58-bit and binary models demonstrate extreme parameter compression, maintaining dense-model reasoning capabilities and complex multi-step logical deduction without accuracy drop remains challenging.
- **Eviction-Aware Reasoning Robustness:** Long-context KV cache eviction policies frequently discard subtle contextual instructions, leading to instruction-following failures and security vulnerabilities such as system prompt leakage [4]. Developing semantic-aware retention mechanisms is an open frontier.
- **Hardware-Algorithm Co-Design:** Standard GPU architectures are heavily optimized for floating-point MAC operations. Fully realizing the theoretical speedups of ternary arithmetic (BitNet b1.58) and sparse low-bit representations requires native hardware accelerators with custom integer addition and bit-packing pipelines [1][3].
- **Dynamic Speculative Overhead:** While multi-branch draft trees (EAGLE-2, Medusa) accelerate decoding, managing draft-generation overhead during highly unpredictable, creative text generation remains difficult [15][7].

## References
[1] A Survey of Low-bit Large Language Models: Basics, Systems, and Algorithms. arxiv. https://arxiv.org/abs/2409.16694 (2024-09-25)
[2] Small Language Models: Survey, Measurements, and Insights. arxiv. https://arxiv.org/abs/2409.15790 (2025-02-26)
[3] The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. arxiv. https://arxiv.org/abs/2402.17764 (2024-02-27)
[4] Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization. web. https://arxiv.org/abs/2607.08057 (2026-07-09)
[5] ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference. arxiv. https://arxiv.org/abs/2410.21465 (2024-10-28)
[6] Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. web. https://arxiv.org/abs/2401.10774 (2024-06-14)
[7] EAGLE-2: Faster Inference of Language Models with Context-Aware Dynamic Draft Trees. web. https://arxiv.org/abs/2406.16858 (2024-06-24)
[8] From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving. web. https://arxiv.org/abs/2607.02574 (2026-06-30)
[9] ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization. arxiv. https://arxiv.org/abs/2502.02631 (2025-02-04)
[10] A Comprehensive Survey of Small Language Models in the Era of Large Language Models. arxiv. https://arxiv.org/abs/2411.03350 (2024-12-28)
[11] TinyLlama: An Open-Source Small Language Model. hf-search. https://huggingface.co/papers/2401.02385 (2024-01-04)
[12] SpeCache: Speculative Key-Value Caching for Efficient Generation of LLMs. hf-search. https://huggingface.co/papers/2503.16163 (2025-03-20)
[13] Training Parallel Speculative Draft Models by Directly Minimizing Expected Decoding Rounds. arxiv. https://arxiv.org/abs/2610.10411 (2026-10-07)
[14] Nucleus Speculative Decoding: Plausibility-Aware Verification Beyond Exact Distribution. arxiv. https://arxiv.org/abs/2610.07822 (2026-10-06)
[15] APEX: Speculate smarter, not deeper. arxiv. https://arxiv.org/abs/2610.07780 (2026-10-06)
[16] DLoop: Looped Speculative Decoding. arxiv. https://arxiv.org/abs/2610.07659 (2026-10-06)
[17] Demystifying Small Language Models for Edge Deployment. web. https://aclanthology.org/2025.acl-long.718.pdf (2025-07-01)
[18] LLM Pruning and Distillation in Practice: The Minitron Approach. hf-search. https://huggingface.co/papers/2408.11796 (2024-08-21)
