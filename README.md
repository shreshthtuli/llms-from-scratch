---
layout: default
title: "LLMs from Scratch: A Practical Course"
description: "A hands-on curriculum to build, train, and align large language models from the ground up."
---

# LLMs from Scratch: A Practical Course

![LLM illustration](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-hero.png)

<a target="_blank" href="https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2Fshreshthtuli%2Fllms-from-scratch%2F">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open in Studio" />
</a>

Welcome to **LLMs from Scratch**, an all-killer-no-filler curriculum that takes you from tokenization to alignment with meticulously crafted Jupyter notebooks, actionable theory, and production-ready code. Whether you are a researcher, engineer, or curious builder, this course gives you the scaffolding to demystify modern LLMs and deploy your own.

---

## 🚀 Course Highlights
- **Hands-on notebooks** for every lesson—clone locally or launch instantly in [Lightning Studio](#hands-on-playground).
- **Practical checkpoints** and datasets so you can experiment without babysitting boilerplate.
- **Theory, references, and best practices** interwoven with code so every concept sticks.
- **Production-aware workflow** covering training, scaling, alignment, quantization, and deployment-friendly fine-tuning.

---

## 📚 Course Structure
Each module is a standalone notebook packed with explanations, exercises, and implementation details. View them on GitHub, launch them via GitHub Pages, or open them interactively in Lightning Studio.

| Module | Topic | Notebook |
| --- | --- | --- |
| 01 | Tokenization Foundations | [01-tokenization.ipynb](https://nbviewer.org/github/shreshthtuli/llms-from-scratch/blob/main/01-tokenization.ipynb) |
| 02 | Building a Tiny LLM | [02-tinyllm.ipynb](https://nbviewer.org/github/shreshthtuli/llms-from-scratch/blob/main/02-tinyllm.ipynb) |
| 03 | Advancing Our LLM | [03-advancing-our-llm.ipynb](https://nbviewer.org/github/shreshthtuli/llms-from-scratch/blob/main/03-advancing-our-llm.ipynb) |
| 04 | Data Engineering for LLMs | [04-data.ipynb](https://nbviewer.org/github/shreshthtuli/llms-from-scratch/blob/main/04-data.ipynb) |
| 05 | Scaling Laws in Practice | [05-scaling-laws.ipynb](https://nbviewer.org/github/shreshthtuli/llms-from-scratch/blob/main/05-scaling-laws.ipynb) |
| 06 | Pretraining at Scale | [06-pretraining.ipynb](https://nbviewer.org/github/shreshthtuli/llms-from-scratch/blob/main/06-pretraining.ipynb) |
| 07 | Supervised Fine-Tuning | [07-supervised-finetuning.ipynb](https://nbviewer.org/github/shreshthtuli/llms-from-scratch/blob/main/07-supervised-finetuning.ipynb) |
| 08 | RLHF and Alignment | [08-rlhf-alignment.ipynb](https://nbviewer.org/github/shreshthtuli/llms-from-scratch/blob/main/08-rlhf-alignment.ipynb) |
| 09 | LoRA & RLVR Techniques | [09-lora-rlvr.ipynb](https://nbviewer.org/github/shreshthtuli/llms-from-scratch/blob/main/09-lora-rlvr.ipynb) |
| 10 | Pruning & Distillation | [10-pruning-distillation.ipynb](https://nbviewer.org/github/shreshthtuli/llms-from-scratch/blob/main/10-pruning-distillation.ipynb) |
| 11 | Appendix: Position Embeddings | [11-appendix-position-embeddings.ipynb](https://nbviewer.org/github/shreshthtuli/llms-from-scratch/blob/main/11-appendix-position-embeddings.ipynb) |
| 12 | Appendix: Quantisation Strategies | [12-appendix-quantisation.ipynb](https://nbviewer.org/github/shreshthtuli/llms-from-scratch/blob/main/12-appendix-quantisation.ipynb) |
| 13 | Appendix: Parameter-Efficient Tuning | [13-appendix-peft.ipynb](https://nbviewer.org/github/shreshthtuli/llms-from-scratch/blob/main/13-appendix-peft.ipynb) |

> 💡 **Tip:** GitHub Pages will render this README as a polished landing page. All notebook links above open directly in NbViewer for an optimal in-browser experience. Feel free to fork the repo and enable Pages to publish your own branded course site.

---

## 🧠 What You'll Learn
- The end-to-end data flow of an LLM—from tokenization and batching to inference-time decoding.
- How to implement core transformer components, attention variations, and optimization tricks.
- Strategies for scaling datasets, managing checkpoints, and monitoring training stability.
- Practical alignment techniques: SFT, preference modeling, RLHF, and reward modeling.
- Deployment-ready compression: pruning, distillation, quantization, and PEFT recipes.

---

## ⚙️ Quick Start

### Option A: Launch in Lightning Studio (no setup!)
1. Click the **Open in Studio** badge above.
2. Authenticate with Lightning (or create a free account).
3. Explore the notebooks in a fully provisioned environment with GPU options.

### Option B: Run Locally
1. **Clone the repository**
   ```bash
   git clone https://github.com/shreshthtuli/llms-from-scratch.git
   cd llms-from-scratch
   ```
2. **Install dependencies** (recommended: Python 3.10+)
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch Jupyter**
   ```bash
   jupyter lab
   ```
4. Open any notebook to start experimenting.

> Need data? Check the [`data/`](data/) directory and follow the dataset preparation steps inside each notebook.

---

## 🧭 Suggested Learning Path
1. **Foundations (Modules 01–03)** – Understand tokens, build your first transformer, and iterate on architecture improvements.
2. **Data & Scaling (Modules 04–06)** – Curate corpora, tune training loops, and scale pretraining experiments responsibly.
3. **Alignment (Modules 07–09)** – Apply SFT, RLHF, and efficient adaptation techniques to align your model with human intent.
4. **Optimization (Modules 10–13)** – Compress, fine-tune, and deploy models using state-of-the-art efficiency tricks.
5. **Capstone** – Combine your learnings to train, align, and ship a bespoke LLM tailored to your use case.

Mix and match as needed—every notebook is designed to stand on its own, but following this order unlocks the smoothest learning curve.

---

## 🛠 Hands-On Playground
- **Lightning Studio**: Run the entire repo in the cloud with zero setup using the badge above.
- **GitHub Codespaces**: Launch a dev container directly from the repo for quick edits.
- **Local GPUs / Clusters**: Scripts in [`src/`](src/) support distributed and mixed-precision training out of the box.

---

## 👨‍🏫 About the Instructor
I’m **Shreshth Tuli**—researcher, builder, and educator focused on making advanced ML systems approachable. I’ve shipped production LLMs, authored peer-reviewed papers, and taught hundreds of practitioners how to wield these models responsibly. Expect honest takes, transparent trade-offs, and plenty of real-world war stories.

Connect with me on [Twitter](https://twitter.com/shreshthtuli) · [LinkedIn](https://www.linkedin.com/in/shreshthtuli/) · [Personal Site](https://shreshthtuli.com)

---

## 🤝 Contributions
Contributions, bug reports, and suggestions are warmly welcomed! To contribute:
1. Fork the repo and create a feature branch.
2. Open a PR describing your changes and the motivation behind them.
3. Tag any relevant notebooks or scripts and include screenshots/metrics if applicable.

Check the issue tracker for bite-sized tasks or open a discussion if you want to propose new modules.

---

## 📄 License
This project is open-sourced under the [MIT License](LICENSE). Feel free to use the materials for your own learning, workshops, or derivative courses—just keep attribution intact.

---

Ready to build? Clone the repo, launch a notebook, and start crafting your own LLMs from the ground up. 🚀

