# LLM Test Time Compute Strategies

This repository demonstrates different LLM inference time scaling strategies, including:

*   Beam Search
*   Monte Carlo Tree Search (MCTS)
*   GRPO (Generative Pre-trained Transformer with Reinforcement Learning from Preference Optimization)

## Overview

The project explores methods to improve the quality and efficiency of LLM-generated outputs by strategically allocating compute time during inference. It uses different models and leverages Groq for accelerated beam search and MCTS. Training the models for GRPO requires a GPU.

## Repository Structure

*   `beam_search.py`: Implements beam search for LLM inference.
*   `mcts.py`: Implements Monte Carlo Tree Search for LLM inference.
*   `grpo.ipynb`: A Jupyter Notebook demonstrating GRPO training and inference.
*   `requirements.txt`: Lists the Python packages required to run the code.

