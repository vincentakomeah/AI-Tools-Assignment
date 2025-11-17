# AI-Tools-Assignment

: Explain the primary differences between TensorFlow and PyTorch. When would 
you choose one over the other?**
 **Answer:**-**Computation graph model:** PyTorch uses dynamic (eager) computation graphs 
by default, which makes debugging and prototyping intuitive — operations are 
executed immediately. TensorFlow originally used static graphs (define-and-run) 
via `tf.Graph`, making deployment on production or optimized graph transforms 
easier; however, TensorFlow 2.x introduced eager execution to align better with 
PyTorch.-**Ecosystem & deployment:** TensorFlow has a larger edge and production 
deployment ecosystem (TensorFlow Serving, TensorFlow Lite, TensorFlow.js, TPU 
support). PyTorch has improved deployment (TorchScript, TorchServe) but 
TensorFlow historically had an edge for production pipelines.-**Research vs Production:** PyTorch is often preferred in research due to its 
Pythonic API and dynamic graphs. TensorFlow is favored in some production 
contexts and in organizations with existing TF infrastructure.-**APIs & usability:** PyTorch code feels more like standard Python; TensorFlow 
2 with `tf.keras` narrowed the gap and improved usability.
 **When to choose:**-Choose **PyTorch** for fast prototyping, research experiments, and when you 
value an intuitive debugging experience.-Choose **TensorFlow** if you need robust deployment options across mobile/web/
 TPU or want the mature production tooling.
 **Q2: Describe two use cases for Jupyter Notebooks in AI development.**
 **Answer:**
 1. **Exploratory Data Analysis (EDA):** Interactive charts, step-by-step 
transformations, data sampling, and quick visual checks are ideal in notebooks.
 2. **Prototyping models & experiments:** Running small experiments, visualizing 
training curves, and documenting hyperparameter changes inline are practical 
uses.
 **Q3: How does spaCy enhance NLP tasks compared to basic Python string 
operations?**
 **Answer:**-**Robust tokenization:** spaCy tokenizes text using linguistic rules (handling 
punctuation, contractions, multi-word tokens), whereas basic string split is 
naive.-**Pretrained models:** spaCy provides pretrained pipelines for POS tagging, 
6
dependency parsing, and NER, giving structured linguistic annotations.-**Efficiency & production-readiness:** spaCy is optimized for speed and memory 
and offers easy serialization, making it suitable for production NLP pipelines.
 ### 2. Comparative Analysis
 #### Scikit-learn vs TensorFlow-**Target applications:**-**Scikit-learn:** Classical ML — linear models, SVMs, decision trees, 
ensemble methods, clustering, dimensionality reduction.-**TensorFlow:** Deep learning and neural networks (CNNs, RNNs, 
Transformers), large-scale model training on GPUs/TPUs.-**Tools & Resources:**-**Scikit-learn:** Simple API for model selection, pipelines, and evaluation. 
Great for small to medium datasets.-**TensorFlow:** Rich set of tools for deep learning (Keras API, TF Datasets, 
TF Hub), deployment (TF Lite, TF Serving, TF.js), and hardware acceleration.-**Ease of use for beginners:** Scikit-learn is often easier for beginners due 
to its consistent API and smaller conceptual overhead. TensorFlow (especially 
TF2 with Keras) has become much more beginner-friendly.-**Community support:** Both have strong communities; TensorFlow has broader 
adoption in large-scale DL applications, while scikit-learn dominates teaching 
and classical ML tasks.
 #### Frameworks & Platforms-**Frameworks:** TensorFlow, PyTorch, Scikit-learn, spaCy — each covers 
different needs (deep learning, research, classical ML, and NLP respectively).-**Platforms:** Google Colab provides free GPU/TPU access for prototyping; 
Jupyter Notebook is excellent for EDA and experiment logs.-**Datasets:** Kaggle hosts community datasets and competitions; TensorFlow 
Datasets provides easy programmatic access to standard datasets.
 ### Why This Matters-**Real-World Impact:** These tools power solutions across industries — from 
medical imaging to fraud detection.-**Skill Validation:** Employers value hands-on ability with TensorFlow, 
PyTorch, and scikit-learn.
 7
### Practical Tips & Resources- Use official docs: TensorFlow, PyTorch, spaCy, scikit-learn.- Post questions with #AIToolsAssignment on the LMS community.- Pro tip: test small pieces independently and keep notebooks clean with clear 
narrative cells.
 ## Part 2: Practical Implementation (50%)
 ### Task 1: Classical ML with Scikit-learn
 The provided script `ai_tools_assignment.py` (in this submission) performs a 
classical ML workflow using scikit-learn on the Breast Cancer dataset. It trains 
multiple models, evaluates them, saves models and scaler, and supports 
prediction via CLI.
 ### How to run
 1. Create a virtual environment and install requirements:
 ```bash
 python -m venv venv
 source venv/bin/activate  # or venv\Scripts\activate on Windows
 pip install -U pip
 pip install scikit-learn pandas joblib matplotlib
 1. 
Train models:
 python ai_tools_assignment.py--train
 1. 
2. 
3. 
Predict (example):
 After training, open 
models/metadata.json to see the list of features and prepare a comma
separated input with the correct number of features
