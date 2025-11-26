# 🛠 <u>Machine learning</u>

Machine Learning is a branch of Artificial Intelligence (AI) that enables computers to learn patterns and make decisions or predictions from data without being explicitly programmed.

### 🧷 <u>*TYPES OF MACHINE LEARNING*</u>
1. Supervised Learning
2. Unsupervised Learning
3. Semi-Supervised Learning
4. Reinforcement Learinig

---

## 🅰 <u>*SUPERVISED LEARNING*</u>

> Model learns from labeled data (data with input-output pairs).

### 🎯 Goal:
To predict outcomes for new, unseen data based on past labeled examples.

### 📘 Definition:
In supervised learning, the algorithm is trained using a dataset that contains input features (X) and corresponding output labels (Y).
It learns the mapping function f: X → Y.

### 🧩 Examples:
1. Predicting house prices 🏠
2. Email spam detection 📧
3. Disease diagnosis from medical images 🩺
4. Stock price prediction 📊
5. Weather forecasting 🌦
6. Credit card fraud detection 💳
7. Sentiment analysis on reviews 💬
8. Handwritten digit recognition ✍
9. Predicting student grades 🎓

### ⚙ Techniques / Algorithms:

    Supervised Learning
    │
    ├── Regression (Predicting Continuous Output)
    │   ├── Linear Regression
    │   ├── Polynomial Regression
    │   ├── Ridge Regression
    │   ├── Lasso Regression
    │   └── Support Vector Regression (SVR)
    │
    └── Classification (Predicting Categorical Output)
        ├── Logistic Regression
        ├── Decision Tree
        ├── Random Forest
        ├── Support Vector Machine (SVM)
        ├── Naive Bayes
        ├── K-Nearest Neighbors (KNN)
        └── Neural Networks


---
## 🅱 <u>*UNSUPERVISED LEARNING*</u>

> Model learns from unlabeled data (only inputs, no outputs).

### 🎯 Goal:
To find hidden patterns, structures, or relationships in data.

### 📘 Definition:
In unsupervised learning, the model explores the data’s structure without explicit labels, discovering patterns or grouping similar data points.

### 🧩 Examples:

1. Customer segmentation in marketing 👥
2. Grouping news articles by topic 🗞
3. Market basket analysis (product association) 🛒
4. Anomaly detection (fraud, defects)
5. Image compression or clustering 🖼
6. Topic modeling in text data 🧾
7. Social network analysis 🌐
8. Gene sequence clustering 🧬
9. Recommender system (unsupervised embeddings) 🎧

### ⚙ Techniques / Algorithms:

    Unsupervised Learning
    │
    ├── Clustering (Grouping similar data)
    │   ├── K-Means Clustering
    │   ├── Hierarchical Clustering
    │   ├── DBSCAN
    │   └── Mean Shift
    │
    └── Dimensionality Reduction (Simplifying data)
        ├── Principal Component Analysis (PCA)
        ├── t-SNE
        ├── Autoencoders
        └── Singular Value Decomposition (SVD)

---
## © <u>*SEMI-SUPERVISED LEARNING*</u>

> Model uses a mix of labeled and unlabeled data.

### 🎯 Goal:
To improve learning accuracy when obtaining labeled data is expensive or limited.

### 📘 Definition:
Combines the strengths of supervised and unsupervised learning — the model first learns from the few labeled samples, then extracts patterns from the unlabeled ones.

### 🧩 Examples:

1. Web page classification 🌍
2. Medical image labeling with few labeled scans 🩻
3. Speech recognition with limited transcribed audio 🎙
4. Email categorization 📬
5. Fraud detection 💸
6. Protein structure prediction 🧫
7. Sentiment analysis with small labeled datasets 💭
8. Object recognition in photos 📷
9. Customer behavior prediction 👤

### ⚙ Techniques / Algorithms:

    Semi-Supervised Learning
    │
    ├── Self-Training
    ├── Co-Training
    ├── Graph-Based Methods
    ├── Generative Models (e.g., Variational Autoencoders)
    └── Semi-Supervised Support Vector Machines (S3VM)

---
## ▶ <u>*REINFORCEMENT LEARNING (RL)*</u>

> Model learns by interacting with an environment and receiving rewards or penalties.

### 🎯 Goal:
To learn a sequence of actions that maximize cumulative reward over time.


### 📘 Definition:
An agent takes actions in an environment to achieve a goal, receiving feedback (reward or punishment) that helps it learn optimal strategies.

### 🧩 Examples:

1. Game-playing AI (Chess, Go, Atari) 🎮
2. Robotics and autonomous navigation 🤖
3. Self-driving cars 🚗
4. Dynamic pricing systems 💰
5. Traffic signal control 🚦
6. Recommendation systems 🎬
7. Resource allocation in networks 🌐
8. Stock trading bots 📈
9. Industrial process automation ⚙



### ⚙ Techniques / Algorithms:

    Reinforcement Learning
    │
    ├── Model-Free Methods
    │   ├── Q-Learning
    │   ├── Deep Q-Networks (DQN)
    │   ├── SARSA
    │   └── Policy Gradient Methods
    │
    └── Model-Based Methods
        ├── Monte Carlo Tree Search (MCTS)
        ├── Markov Decision Processes (MDP)
        └── Actor-Critic Methods (A2C, DDPG)


---

## 📧 <u>*SELF-SUPERVISED LEARNING (Emerging Type)*</u>

> Model generates its own labels from data itself.

### 🎯 Goal:
To enable learning from large amounts of unlabeled data efficiently.

### 📘 Definition:
Self-supervised learning uses parts of the input as supervision for other parts (e.g., predicting missing words or image patches).

### 🧩 Examples:

1. Predicting missing words in a sentence (e.g., BERT) 🧠
2. Predicting next frame in a video 🎥
3. Colorizing black-and-white images 🖤➡🎨
4. Predicting rotation of images 🌀
5. Sentence embedding generation 🗣
6. Masked autoencoding 🧩
7. Contrastive learning (SimCLR, MoCo) ⚡
8. Speech representation learning 🎧
9. Code completion in IDEs 💻

### ⚙ Techniques / Algorithms:

    Self-Supervised Learning
    │
    ├── Contrastive Learning (SimCLR, MoCo)
    ├── Masked Language Modeling (BERT)
    ├── Predictive Coding
    ├── Autoencoders
    └── Generative Models (GPT, Vision Transformers)


---

## 🌲 <u>Summary Tree (Condensed Overview)</u>

       MACHINE LEARNING
       │
       ├── Supervised Learning
       │   ├── Regression → (Linear, Lasso, Ridge, SVR)
       │   └── Classification → (Decision Tree, SVM, KNN, RF)
       │
       ├── Unsupervised Learning
       │   ├── Clustering → (K-Means, DBSCAN, Hierarchical)
       │   └── Dimensionality Reduction → (PCA, t-SNE, Autoencoders)
       │
       ├── Semi-Supervised Learning
       │   └── (Self-Training, Co-Training, S3VM, Graph Models)
       │
       ├── Reinforcement Learning
       │   ├── Model-Free → (Q-Learning, DQN, SARSA)
       │   └── Model-Based → (MDP, Actor-Critic, MCTS)
       │
       └── Self-Supervised Learning
       └── (BERT, GPT, SimCLR, Autoencoders)

       
