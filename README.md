# 🚀 Awesome-LLM-Prod

A curated collection of open-source **Large Language Model (LLM)** projects that are **production-ready** and can be used for solving real-world problems. This repository focuses on high-performance, scalable LLM solutions across various industries and applications.

## 📚 Table of Contents
- [🌟 Introduction](#introduction)
- [🎯 Purpose](#purpose)
- Awesome Lists
    - [🧠 Large Language Models](#1-large-language-models)
    - [🛠️ Production Tools](#2-production-tools)
    - [💼 Real-World Applications](#3-real-world-applications)
    - [🔍 Vector Databases and Embeddings](#4-vector-databases-and-embeddings)
    - [🔄 Data Generation, Processing and Management](#5-data-generation-processing-and-management)
- [Contributing](#contributing)
- [License](#license)

## Introduction
With the rise of LLMs in various domains, there is a growing need for solutions that are ready for deployment in production environments. **Awesome-LLM-Prod** aims to provide a collection of **open source, production-grade** LLM repositories, tested and proven to scale, for real-world use cases. Whether you're deploying a large model for NLP tasks or integrating AI into a customer-facing product, this repository offers the tools and frameworks needed for real-world scenarios.

## Purpose
The purpose of this repository is to:
1. Curate open-source LLM projects that are ready for **production environments**.
2. Showcase **real-world applications** of LLMs across various industries.
3. Provide solutions that focus on **scalability, optimization, and deployment**.
4. Bridge the gap between **research prototypes** and **production-grade projects**.


## 1. **Large Language Models**
   - Production-ready LLM projects and implementations.
   - Fine-tuning LLMs for specific tasks.

   |Project Name|Support|Tags|Description|
   |------------|-------|----|-----------|
   |[Axolotl](https://github.com/axolotl-ai-cloud/axolotl)|Community|Training, Fine-Tuning|Tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures|
   |[DeepSpeed](https://github.com/microsoft/DeepSpeed)|Microsoft|Training, Inference, Compression|An optimization library that makes distributed training and inference easy
   |[Hugging Face Transformers](https://github.com/huggingface/transformers)|Hugging Face|Training, Fine-Tuning, Inference, NLP|State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX|
   |[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)|Community|Training, Fine-Tuning|Unified Efficient Fine-Tuning of 100+ LLMs|
   |[LitGPT](https://github.com/Lightning-AI/litgpt)|Lightning-AI|Training, Fine-Tuning, Deployment, Chatbots|20+ high-performance LLMs with recipes to pretrain, finetune and deploy at scale|
   |[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)|NVIDIA|Training, Fine-Tuning|GPU optimized techniques for training transformer models at-scale|
   |[ONNX Runtime](https://github.com/microsoft/onnxruntime)|Microsoft|Inference, Training-Optimization|Cross-platform, high performance ML inferencing and training accelerator|


## 2. **Production Tools**
   - Tools for inference, evaluating, testing, monitoring, and scaling LLMs.
   - Deployment solutions for cloud and edge environments.
   - Optimization techniques to reduce memory usage, latency, and costs.

|Project Name|Support|Tags|Description|
|------------|-------|----|-----------|
|[BentoML](https://github.com/bentoml/BentoML)|BentoML|RAG, Model-Serving, API, Deployment|Framework for serving, managing, and deploying machine learning models|
|[LitServe](https://github.com/Lightning-AI/LitServe)|Lightning.AI|Inference, Model-Serving, Deployment|Lightning-fast serving engine for AI models|
|[LMDeploy](https://github.com/InternLM/lmdeploy)|InternLM|Inference, Deployment, Optimization|A toolkit for compressing, deploying, and serving LLM with high performance and low latency|
|[MLflow](https://github.com/mlflow/mlflow)|Databricks|Experiment Tracking, Model Registry, Deployment|An open source platform for the machine learning lifecycle|
|[OpenVINO](https://github.com/openvinotoolkit/openvino)|Intel|Inference, Optimization, Deployment|Toolkit for optimizing and deploying AI models across Intel hardware|
|[Ray](https://github.com/ray-project/ray)|Anyscale|Distributed Computing, Scaling, Inference, Deployment|A unified framework for scaling AI and Python applications|
|[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)|NVIDIA|Inference, Optimization|Optimize and deploy LLMs on NVIDIA GPUs|
|[Triton Inference Server](https://github.com/triton-inference-server/server)|NVIDIA|Model-Serving, Inference, Deployment|Optimized and production-ready model inference server|
|[vllm](https://github.com/vllm-project/vllm)|vllm-project|Inference, Deployment, Model-Serving|A high-throughput and memory-efficient inference and serving engine for LLMs|
|[Weights & Biases](https://wandb.ai/site)|Weights & Biases|Experiment Tracking, Visualization, Collaboration|MLOps platform for tracking experiments and managing machine learning projects|

## 3. **Real-World Applications**
   - App Enablers
   - Prompt optimizations
   - Structured Output
   - Projects applying LLMs to healthcare, finance, customer service, and other industries.

|Project Name|Support|Tags|Description|
|------------|-------|----|-----------|
|[AdalFlow](https://github.com/SylphAI-Inc/AdalFlow)|SylphAI-Inc|RAG, Agents, LLM Eval, Trainers, Optimizers|The library to build & auto-optimize any LLM task|
|[DSPy](https://github.com/stanfordnlp/dspy)|StanfordNLP|RAG, Prompt-Optimization, Information-Extraction|Framework for programming—not prompting—foundation models|
|[Guidance](https://github.com/microsoft/guidance)|Microsoft|Templating, Generation-Control, Structured-Output|A guidance language for controlling LLMs|
|[Haystack](https://github.com/deepset-ai/haystack)|deepset-ai|RAG, Question-Answering, Information-Retrieval|End-to-end NLP framework for building applications powered by LLMs and Transformer models|
|[LangChain](https://github.com/langchain-ai/langchain)|langchain-ai|RAG, Structured-Output, Chatbots, Agents|LangChain is a framework for developing applications powered by LLMs|
|[LlamaIndex](https://github.com/jerryjliu/llama_index)|Community|RAG, Data-Ingestion, Structured-Data|Data Framework for LLM applications to ingest, structure, and access private or domain-specific data|
|[mem0](https://github.com/mem0ai/mem0)|mem0ai|Memory-Layer|Enhances AI assistants and agents with an intelligent memory layer|
|[outlines](https://github.com/dottxt-ai/outlines)|dottxt-ai|Structured-Output|Library for Structured Text Generation|
|[Semantic Kernel](https://github.com/microsoft/semantic-kernel)|Microsoft|AI-Orchestration, Plugins, Connectors, AI-services|Integrate cutting-edge LLM technology quickly and easily into your apps|
|[TTS](https://github.com/coqui-ai/TTS)|coqui-ai|Text-to-Speech|a deep learning toolkit for Text-to-Speech, battle-tested in research and production|

## 4. **Vector Databases and Embeddings**
   - Vector databases for efficient similarity search.
   - Embedding tools for text-to-vector conversion.
   - Indexing and retrieval solutions for large-scale datasets.

|Project Name|Support|Tags|Description|
|------------|-------|----|-----------|
|[Faiss](https://github.com/facebookresearch/faiss)|Facebook Research|Vector-Database, Similarity-Search|A library for efficient similarity search and clustering of dense vectors|
|[Milvus](https://github.com/milvus-io/milvus)|Zilliz|Vector-Database|An open-source vector database built to power embedding similarity search|
|[Pinecone](https://www.pinecone.io/)|Pinecone|Vector-Database|Managed vector database for machine learning applications|
|[Qdrant](https://github.com/qdrant/qdrant)|Qdrant|Vector-Database, Rust|Vector similarity search engine and database|
|[sentence-transformers](https://github.com/UKPLab/sentence-transformers)|UKPLab|Embeddings, Fine-Tuning, Multilingual|Provides an easy method to compute dense vector representations for sentences, paragraphs, and images|
|[Weaviate](https://github.com/weaviate/weaviate)|SeMI Technologies|Vector-Database, GraphQL|Open source vector database that stores both objects and vectors|

## 5. **Data Generation, Processing and Management**
   - Tools for data generation, cleaning, preprocessing, and augmentation.
   - Data versioning and lineage tracking solutions.
   - High-quality datasets for training and fine-tuning LLMs in production environments.rew

|Project Name|Support|Tags|Description|
|------------|-------|----|-----------|
|[Argilla](https://github.com/argilla-io/argilla)|Argilla-IO|Data-Generation, Data-Quality|collaboration tool for AI engineers and domain experts to build high-quality datasets|
|[DVC (Data Version Control)](https://github.com/iterative/dvc)|Iterative|Data-Versioning, ML-Pipelines|Open-source version control system for machine learning projects|
|[Dolt](https://github.com/dolthub/dolt)|DoltHub|Data-Versioning, SQL-Database|Git for data: Version control system for structured data|
|[NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator)|NVIDIA|Data-Generation, Data-Processing, Scalability|Scalable data pre processing and curation toolkit for LLMs|
|[Pachyderm](https://github.com/pachyderm/pachyderm)|Pachyderm|Data-Versioning, Data-Pipelines|Data-Centric Pipelines and Data Versioning|
|[Snorkel](https://github.com/snorkel-team/snorkel)|Snorkel AI|Data -Labeling, Weak-Supervision|A system for programmatically building and managing training datasets|


** Note that some of the projects have overlapping categories, but have been classified based on intuitive understanding. If you think a different category better suits a project, please feel free to open a PR.

## Contributing
We welcome contributions from the community! If you know of any production-grade LLM project that fits our criteria, please feel free to open a pull request.

## License
This repository is dedicated to the public domain under the Creative Commons CC0 1.0 Universal license. For more details, see the `LICENSE` file or visit [https://creativecommons.org/publicdomain/zero/1.0/](https://creativecommons.org/publicdomain/zero/1.0/).
