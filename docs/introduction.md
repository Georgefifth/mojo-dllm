<div class="hero">
  <img src="images/logo.svg" alt="mojo-dllm logo: masked tokens resolving into a flame">
  <div class="tagline">Local inference for diffusion language models, written in Mojo.</div>
</div>

# Introduction

mojo-dllm runs **LLaDA-8B** and **Dream-7B**, two diffusion language models,
from quantized GGUF files on your own CPU or on an NVIDIA GPU. The whole
inference path is Mojo: the GGUF reader, the tokenizer, the quantized matrix
multiply, the GPU kernels, the transformer and the denoising loop. You need
no Python at runtime, and the GPU is optional.

A diffusion language model writes differently from the models you know. An
autoregressive model appends one token at a time. LLaDA starts from a canvas
of `[MASK]` tokens and fills it in over a fixed number of steps, committing the
tokens it is most sure about first. [Diffusion decoding in five
minutes](diffusion.md) walks through it with a picture.

## What works today

- `mojo-dllm run` generates text from LLaDA-8B-Instruct and
  Dream-v0-Instruct-7B (Q4_K_M) on x86 CPUs with AVX2, using AVX-VNNI when
  present.
- `--device gpu` runs both models on NVIDIA GPUs with compute capability 8.0
  or newer. It has been measured on an RTX 3050 Laptop GPU only.
- Both tokenizers match llama.cpp id for id on a 36-case multilingual corpus.
- Logits match llama.cpp's implementation of both models; the measured
  agreement is in [Correctness](correctness.md).
- [Benchmarks](benchmarks.md) against llama.cpp and diffuse-cpp on the same
  machine, with every number generated from a result file in the repository.

## What does not work yet

- The GPU path is measured on one laptop part, an RTX 3050 held to 30 W,
  against llama.cpp's CUDA build; [Benchmarks](benchmarks.md) has the numbers.
  It supports NVIDIA only.
- Two architectures: LLaDA and Dream. Others stop with
  `unsupported model architecture: <name> (supported: llada, dream)`.
- One quantization mix: the F32, Q4_K and Q6_K tensors a Q4_K_M file
  contains. Any other tensor type stops the load with an error that names it.

## Where to go next

- Run it: [Quick start](quickstart.md).
- Read the numbers: [Benchmarks](benchmarks.md).
- Read the code: [Architecture](architecture.md), then the source under
  `src/mojo_dllm/`.
