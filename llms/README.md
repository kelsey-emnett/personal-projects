# Building Character Biographies with LLMs

## Overview
This code builds a RAG application using five Alexendre Dumas books. The user provides a character from one of these books and the RAG application returns a short character biography.

## Environment
This code was built in Databricks and uses Databricks 14.3 ML Runtime. Code linting was run with pre-commit and ruff. Python 3 was used. Langchain was used for RAG development. Evaluation was done using Trulens. This was run on an 8 core, 56 GB RAM single-node cluster.

## Project Goals
- Build a RAG application to build informative character biographies with important events in the plot described
- Have biographies grounded with relevant context from the provided texts rather than pre-existing Foundation Model knowledge

## Project Features:
- Uses Langchain semantic chunker to chunk book text
- Used Langchain prompt templates to build a complex, dynamic prompt
- Used Databricks Fondation Models for LLMs and embedding models
- Used the Trulens RAG triad to evaluate the LLM