# Awesome LLM Prod [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated collection of open-source Large Language Model (LLM) projects that are production-ready and can be used for solving real-world problems.

With the rise of LLMs in various domains, there is a growing need for solutions that are ready for deployment in production environments. This list curates open-source, production-grade LLM repositories that are tested and proven to scale, bridging the gap between research prototypes and production-grade projects. Whether you're deploying a large model for NLP tasks or integrating AI into a customer-facing product, these are the tools and frameworks for real-world scenarios.

## Contents

- [Large Language Models](#large-language-models)
- [Production Tools](#production-tools)
- [Evaluation and Observability](#evaluation-and-observability)
- [Agents and Orchestration](#agents-and-orchestration)
- [Real-World Applications](#real-world-applications)
- [Vector Databases and Embeddings](#vector-databases-and-embeddings)
- [Data Generation, Processing and Management](#data-generation-processing-and-management)

## Large Language Models

Production-ready projects for training, fine-tuning, and post-training LLMs.

| Project Name                                                             | Support              | Tags                                        | Description                                                                                                                      |
| ------------------------------------------------------------------------ | -------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)                   | Community            | Training, Fine-Tuning                       | Tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures |
| [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)                    | Microsoft            | Training, Inference, Compression            | An optimization library that makes distributed training and inference easy                                                       |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Hugging Face         | Training, Fine-Tuning, Inference, NLP       | State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX                                                               |
| [LitGPT](https://github.com/Lightning-AI/litgpt)                         | Lightning-AI         | Training, Fine-Tuning, Deployment, Chatbots | 20+ high-performance LLMs with recipes to pretrain, finetune and deploy at scale                                                 |
| [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)                 | Community            | Training, Fine-Tuning                       | Unified Efficient Fine-Tuning of 100+ LLMs                                                                                       |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)                     | NVIDIA               | Training, Fine-Tuning                       | GPU optimized techniques for training transformer models at-scale                                                                |
| [ms-swift](https://github.com/modelscope/ms-swift)                       | ModelScope (Alibaba) | Fine-Tuning, PEFT, Multimodal               | Framework for fine-tuning (CPT/SFT/DPO/GRPO) 600+ LLMs and 300+ multimodal models                                                |
| [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)                             | NVIDIA               | Post-Training, Fine-Tuning, GRPO, DPO       | Scalable and efficient post-training library for RL                                                                              |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime)                 | Microsoft            | Inference, Training-Optimization            | Cross-platform, high performance ML inferencing and training accelerator                                                         |
| [PEFT](https://github.com/huggingface/peft)                              | Hugging Face         | PEFT, LoRA, Fine-Tuning                     | State-of-the-art parameter-efficient fine-tuning; the adapter layer under most production LoRA workflows                         |
| [PyLate](https://github.com/lightonai/pylate)                            | LightOn              | Late-Interaction, Retrieval, Fine-Tuning    | Flexible training and retrieval for late-interaction (ColBERT-style) models, built on Sentence Transformers                      |
| [TRL](https://github.com/huggingface/trl)                                | Hugging Face         | Post-Training, RLHF, Fine-Tuning            | Full-stack library to post-train transformers: SFT, DPO, GRPO, PPO and reward modeling                                           |
| [Unsloth](https://github.com/unslothai/unsloth)                          | Unsloth AI           | Fine-Tuning, LoRA, Quantization             | Fast, memory-efficient fine-tuning and RL for open LLMs via LoRA/QLoRA                                                           |
| [verl](https://github.com/verl-project/verl)                             | ByteDance Seed       | RLHF, Post-Training, Distributed-Training   | Flexible and production-ready RL post-training framework for LLMs (HybridFlow)                                                   |


## Production Tools

Tools for inference, serving, deployment, and scaling of LLMs in cloud and edge environments, including optimization techniques to reduce memory usage, latency, and costs.

| Project Name                                                                 | Support      | Tags                                                  | Description                                                                                                |
| ---------------------------------------------------------------------------- | ------------ | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [BentoML](https://github.com/bentoml/BentoML)                                | BentoML      | RAG, Model-Serving, API, Deployment                   | Framework for serving, managing, and deploying machine learning models                                     |
| [KServe](https://github.com/kserve/kserve)                                   | CNCF         | Model-Serving, Kubernetes, Deployment                 | Standardized generative and predictive model inference platform on Kubernetes                              |
| [LiteLLM](https://github.com/BerriAI/litellm)                                | BerriAI      | Gateway, Routing, Cost-Tracking                       | Call 100+ LLM providers through one OpenAI-format API/proxy with cost tracking, fallbacks, and rate limits |
| [LitServe](https://github.com/Lightning-AI/LitServe)                         | Lightning.AI | Inference, Model-Serving, Deployment                  | Lightning-fast serving engine for AI models                                                                |
| [llama.cpp](https://github.com/ggml-org/llama.cpp)                           | ggml.ai      | Inference, Edge, Quantization                         | LLM inference in pure C/C++ with GGUF quantization; runs everywhere from servers to phones                 |
| [LMDeploy](https://github.com/InternLM/lmdeploy)                             | InternLM     | Inference, Deployment, Optimization                   | A toolkit for compressing, deploying, and serving LLM with high performance and low latency                |
| [MLflow](https://github.com/mlflow/mlflow)                                   | Databricks   | Experiment Tracking, Model Registry, Deployment       | An open source platform for the machine learning lifecycle                                                 |
| [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)                         | NVIDIA       | Inference, Distributed, Model-Serving                 | Datacenter-scale distributed inference with disaggregated prefill/decode and KV-aware routing              |
| [Ollama](https://github.com/ollama/ollama)                                   | Ollama       | Model-Serving, Local-LLM, Deployment                  | Run LLMs locally with simple model packaging and an OpenAI-compatible API                                  |
| [OpenVINO](https://github.com/openvinotoolkit/openvino)                      | Intel        | Inference, Optimization, Deployment                   | Toolkit for optimizing and deploying AI models across Intel hardware                                       |
| [Ray](https://github.com/ray-project/ray)                                    | Anyscale     | Distributed Computing, Scaling, Inference, Deployment | A unified framework for scaling AI and Python applications                                                 |
| [SGlang](https://github.com/sgl-project/sglang)                              | Community    | Inference, Model-Serving, Deployment, VLMs            | SGLang is a fast serving framework for large language models and vision language models                    |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)                       | NVIDIA       | Inference, Optimization                               | Optimize and deploy LLMs on NVIDIA GPUs                                                                    |
| [Triton Inference Server](https://github.com/triton-inference-server/server) | NVIDIA       | Model-Serving, Inference, Deployment                  | Optimized and production-ready model inference server                                                      |
| [vllm](https://github.com/vllm-project/vllm)                                 | vllm-project | Inference, Deployment, Model-Serving                  | A high-throughput and memory-efficient inference and serving engine for LLMs                               |

## Evaluation and Observability

Evaluation frameworks, benchmarking, automated red-teaming, and production tracing and monitoring for LLM apps.

| Project Name                                                                 | Support    | Tags                                      | Description                                                                                  |
| ---------------------------------------------------------------------------- | ---------- | ----------------------------------------- | -------------------------------------------------------------------------------------------- |
| [Langfuse](https://github.com/langfuse/langfuse)                             | Langfuse   | Observability, Tracing, Prompt-Management | Open-source LLM engineering platform: tracing, evals, prompt management, and metrics         |
| [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness) | EleutherAI | LLM-Evaluation, Benchmarking              | A framework for few-shot evaluation of language models with 60+ academic benchmarks          |
| [promptfoo](https://github.com/promptfoo/promptfoo)                          | Promptfoo  | LLM-Evaluation, Red-Teaming, Testing      | Test-driven LLM development: evals, benchmarking, and automated red-teaming for apps and RAG |

## Agents and Orchestration

Agent frameworks, multi-agent orchestration, and agent memory infrastructure.

| Project Name                                                        | Support      | Tags                                               | Description                                                                                        |
| ------------------------------------------------------------------- | ------------ | -------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| [CrewAI](https://github.com/crewAIInc/crewAI)                       | CrewAI       | Agents, Multi-Agent, Orchestration                 | Framework for orchestrating role-playing, autonomous AI agents working together as crews           |
| [LangGraph](https://github.com/langchain-ai/langgraph)              | langchain-ai | Agents, Orchestration, Stateful-Workflows          | Low-level stateful orchestration framework for building production agents                          |
| [mem0](https://github.com/mem0ai/mem0)                              | mem0ai       | Memory-Layer                                       | Enhances AI assistants and agents with an intelligent memory layer                                 |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) | OpenAI       | Agents, Multi-Agent, MCP                           | OpenAI's official agent framework with handoffs, guardrails, sessions, and first-class MCP support |
| [PydanticAI](https://github.com/pydantic/pydantic-ai)               | Pydantic     | Agents, Structured-Output, Type-Safety             | Type-safe, model-agnostic agent framework with validated outputs, from the Pydantic team           |
| [Semantic Kernel](https://github.com/microsoft/semantic-kernel)     | Microsoft    | AI-Orchestration, Plugins, Connectors, AI-services | Integrate cutting-edge LLM technology quickly and easily into your apps                            |

## Real-World Applications

App enablers applying LLMs to real-world problems: prompt optimization, structured output, few-shot classification, document processing, speech, time series, and industry applications.

| Project Name                                                        | Support                  | Tags                                                       | Description                                                                                                                              |
| ------------------------------------------------------------------- | ------------------------ | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| [AdalFlow](https://github.com/SylphAI-Inc/AdalFlow)                 | SylphAI-Inc              | RAG, Agents, LLM Eval, Trainers, Optimizers                | The library to build & auto-optimize any LLM task                                                                                        |
| [Crawl4AI](https://github.com/unclecode/crawl4ai)                   | Community                | Web-Crawling, Scraping, RAG                                | LLM-friendly web crawler and scraper producing clean Markdown for RAG and agent pipelines                                                |
| [Docling](https://github.com/docling-project/docling)               | LF AI & Data (IBM)       | Document-Processing, PDF-to-Text, RAG                      | Converts PDF, DOCX, PPTX and images into AI-ready structured output with layout and table understanding                                  |
| [DSPy](https://github.com/stanfordnlp/dspy)                         | StanfordNLP              | RAG, Prompt-Optimization, Information-Extraction           | Framework for programming—not prompting—foundation models                                                                                |
| [Guidance](https://github.com/guidance-ai/guidance)                 | Microsoft                | Templating, Generation-Control, Structured-Output          | A guidance language for controlling LLMs                                                                                                 |
| [Haystack](https://github.com/deepset-ai/haystack)                  | deepset-ai               | RAG, Question-Answering, Information-Retrieval             | End-to-end NLP framework for building applications powered by LLMs and Transformer models                                                |
| [Instructor](https://github.com/567-labs/instructor)                | 567 Labs                 | Structured-Output, Extraction, Validation                  | Structured LLM outputs with Pydantic validation, retries, and support for 15+ providers                                                  |
| [LangChain](https://github.com/langchain-ai/langchain)              | langchain-ai             | RAG, Structured-Output, Chatbots, Agents                   | LangChain is a framework for developing applications powered by LLMs                                                                     |
| [LlamaIndex](https://github.com/run-llama/llama_index)              | Community                | RAG, Data-Ingestion, Structured-Data                       | Data Framework for LLM applications to ingest, structure, and access private or domain-specific data                                     |
| [Marker](https://github.com/datalab-to/marker)                      | Datalab                  | PDF-to-Text, Document-Processing                           | A tool for converting PDFs to markdown + JSON, enabling document processing and analysis                                                 |
| [MarkItDown](https://github.com/microsoft/markitdown)               | Microsoft                | Document-Processing, Markdown                              | Lightweight utility for converting Office files, PDFs, images, and audio to Markdown for LLM pipelines                                   |
| [outlines](https://github.com/dottxt-ai/outlines)                   | dottxt-ai                | Structured-Output                                          | Library for Structured Text Generation                                                                                                   |
| [SetFit](https://github.com/huggingface/setfit)                     | Hugging Face             | Few-Shot, Classification, Fine-Tuning                      | Efficient few-shot text classification with Sentence Transformers                                                                        |
| [Time-Series-Library](https://github.com/thuml/Time-Series-Library) | THUML                    | Time-Series, Forecasting, Analysis, Classification         | A comprehensive library for deep time series models covering forecasting, imputation, anomaly detection, and classification              |
| [TTS (Coqui fork)](https://github.com/idiap/coqui-ai-TTS)           | Idiap Research Institute | Text-to-Speech                                             | Actively maintained fork of Coqui TTS (original is unmaintained), a deep learning toolkit for Text-to-Speech battle-tested in production |
| [Turftopic](https://github.com/x-tabdeveloping/turftopic)           | x-tabdeveloping          | Topic-Modeling, Text-Classification, Sentence-Transformers | Zero-shot topic modeling and text classification using LLMs                                                                              |

## Vector Databases and Embeddings

Vector databases for efficient similarity search, embedding tools and servers, and large-scale indexing and retrieval.

| Project Name                                                                          | Support           | Tags                                  | Description                                                                                           |
| ------------------------------------------------------------------------------------- | ----------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [Chroma](https://github.com/chroma-core/chroma)                                       | Chroma            | Vector-Database, Embeddings, RAG      | AI-native open-source embedding database, from local development to distributed cloud                 |
| [Faiss](https://github.com/facebookresearch/faiss)                                    | Facebook Research | Vector-Database, Similarity-Search    | A library for efficient similarity search and clustering of dense vectors                             |
| [LanceDB](https://github.com/lancedb/lancedb)                                         | LanceDB           | Vector-Database, Multimodal, Embedded | Embedded, serverless vector database built on the Lance columnar format for multimodal AI             |
| [Milvus](https://github.com/milvus-io/milvus)                                         | Zilliz            | Vector-Database                       | An open-source vector database built to power embedding similarity search                             |
| [pgvector](https://github.com/pgvector/pgvector)                                      | pgvector          | Vector-Database, PostgreSQL           | Vector similarity search as a PostgreSQL extension with exact and ANN (HNSW/IVFFlat) indexes          |
| [Qdrant](https://github.com/qdrant/qdrant)                                            | Qdrant            | Vector-Database, Rust                 | Vector similarity search engine and database                                                          |
| [sentence-transformers](https://github.com/huggingface/sentence-transformers)         | Hugging Face      | Embeddings, Fine-Tuning, Multilingual | Provides an easy method to compute dense vector representations for sentences, paragraphs, and images |
| [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) | Hugging Face      | Embeddings, Inference, Model-Serving  | High-throughput, low-latency server for embedding, reranker, and classifier models                    |
| [Vespa](https://github.com/vespa-engine/vespa)                                        | Vespa.ai          | Vector-Search, Hybrid-Search          | Big-data serving engine combining vector, lexical, and structured search with ML inference at scale   |
| [Weaviate](https://github.com/weaviate/weaviate)                                      | SeMI Technologies | Vector-Database, GraphQL              | Open source vector database that stores both objects and vectors                                      |

## Data Generation, Processing and Management

Tools for data generation, cleaning, labeling, versioning, and lineage tracking for LLM training data.

| Project Name                                                   | Support      | Tags                                          | Description                                                                                   |
| -------------------------------------------------------------- | ------------ | --------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [Argilla](https://github.com/argilla-io/argilla)               | Argilla-IO   | Data-Generation, Data-Quality                 | collaboration tool for AI engineers and domain experts to build high-quality datasets         |
| [DataTrove](https://github.com/huggingface/datatrove)          | Hugging Face | Data-Processing, Deduplication, Pipelines     | Platform-agnostic pipelines to process, filter, and dedup web-scale text data (built FineWeb) |
| [Dolt](https://github.com/dolthub/dolt)                        | DoltHub      | Data-Versioning, SQL-Database                 | Git for data: Version control system for structured data                                      |
| [DVC (Data Version Control)](https://github.com/treeverse/dvc) | Treeverse    | Data-Versioning, ML-Pipelines                 | Open-source version control system for machine learning projects                              |
| [Label Studio](https://github.com/HumanSignal/label-studio)    | HumanSignal  | Data-Labeling, Annotation, RLHF               | Multi-type data labeling and annotation platform with standardized output format              |
| [NeMo-Curator](https://github.com/NVIDIA-NeMo/Curator)         | NVIDIA       | Data-Generation, Data-Processing, Scalability | Scalable data pre processing and curation toolkit for LLMs                                    |
## Contributing

Contributions are welcome! Please read the [contribution guidelines](contributing.md) first: entries must be open source, actively maintained, and proven in production. Some projects fit several categories and are classified by primary use, so feel free to open a PR if you think a different category better suits a project.
