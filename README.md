# CEI_Ayaz_Assignments 

# Week 1: Mathematics & Statistics Foundations for Data Science

## Project Overview
The objective of this first week was to complete foundational mathematical and statistical exercises required for advanced data science workflows. The assignment focused on implementing core principles of Linear Algebra, Descriptive & Inferential Statistics, and Probability Theory within a Jupyter Notebook environment.

## Key Implementation Concepts
* **Linear Algebra:** Implemented matrix operations, vector spaces, determinants, eigenvalues, and eigenvectors, which form the mathematical backbone of data transformations and model training.
* **Statistics:** Analyzed data distributions using measures of central tendency (mean, median, mode), dispersion (variance, standard deviation), and verified properties of the Normal Distribution.
* **Probability Theory:** Applied probability concepts, conditional probability, and Bayes' Theorem to understand risk, prediction modeling, and predictive uncertainty.

## Technical Stack
* **Language:** Python
* **Libraries:** NumPy, SciPy, Matplotlib, Seaborn

## Key Learnings
* Solidified the mathematical intuition behind how machine learning algorithms optimize weights and cost functions.
* Mastered using NumPy and SciPy to programmatically solve linear equations and statistically analyze distribution variance.



# Week 2: End-to-End Predictive ML Pipeline on Sales Data

## Project Overview
The objective of this assignment was to design and implement an end-to-end Machine Learning pipeline using historical Tesla production and delivery data (2015-2025). The project involved transforming raw data into reliable forecasting features and evaluating predictive model capabilities.

## Key Implementation Steps
* **Exploratory Data Analysis (EDA):** Analyzed historical production and delivery variables to uncover distribution shapes, anomalies, and underlying growth trajectories.
* **Feature Engineering & Preprocessing:** Handled missing indices, scaled features, and engineered rolling historical averages to structure the data for temporal tracking.
* **Regression Modeling & Tuning:** Built and optimized regression algorithms using hyperparameter tuning to accurately capture market patterns and minimize forecasting errors.
* **Time Series Forecasting:** Extended the pipeline structure to handle time-dependent patterns, projecting future delivery and production numbers over sequential intervals.

## Technical Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

## Key Learnings
* Gained hands-on experience structuring a complete, production-grade ML pipeline from ingestion to prediction.
* Mastered the nuances of feature engineering for time-ordered data without introducing data leakage.



# Week 3: Customer Intelligence System using Classification, Ensemble & Clustering

## Project Overview
The objective of this project was to develop a comprehensive Customer Intelligence System by combining unsupervised and supervised learning methodologies. Utilizing complex socio-economic and behavioral metrics, the system identifies distinct target groups and builds predictive models to classify high-value clusters.

## Key Implementation Steps
* **Unsupervised Clustering:** Implemented **K-Means** and **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** to naturally segment data points into behavioral profiles without historical bias.
* **Supervised Classification & Ensemble Learning:** Developed robust classification pipelines using ensemble models like **Random Forest** and **XGBoost** to learn the boundaries of the generated clusters and predict segment classifications.
* **Performance Optimization:** Tuned hyperparameters and evaluated cluster cohesion (via Silhouette analysis) alongside classification metrics (Precision, Recall, F1-Score) to ensure optimized predictive accuracy.

## Technical Stack
* **Language:** Python
* **Libraries:** Scikit-Learn, XGBoost, NumPy, Pandas, Matplotlib, Seaborn

## Key Learnings
* Learned how to marry unsupervised data discoveries with supervised predictive modeling to drive actionable customer segmentation.
* Mastered density-based clustering techniques to effectively isolate noise and outliers from structural data groupings.




# Week 4: Image Classification Model on CIFAR-10

## Project Overview
The objective of this assignment was to construct and evaluate an image classification system on the CIFAR-10 dataset. The project focused on conducting an architectural performance analysis, directly benchmarking a traditional Artificial Neural Network (ANN) against a highly structured Convolutional Neural Network (CNN) across shared training strategies.

## Key Implementation Steps
* **High-Speed Data Ingestion:** Normalized pixel intensities to the [0, 1] range and constructed an optimized `tf.data` pipeline leveraging batching and automated prefetching to maximize hardware performance.
* **ANN Baseline Architecture:** Developed a multi-layer fully-connected network utilizing input flattening to evaluate baseline classification capabilities on raw multi-channel pixel arrays.
* **CNN Architecture:** Engineered a convolutional network using spatial feature extraction blocks (`Conv2D` filters and `MaxPooling2D` reduction layers) to preserve pixel adjacency and capture visual textures.
* **Training Strategy Optimization:** Integrated advanced optimization mechanisms, including **Batch Normalization** for accelerated gradient convergence and structured **Dropout layers** (ranging from 0.25 to 0.50) to mitigate overfitting.

## Technical Stack
* **Language:** Python
* **Frameworks:** TensorFlow, Keras, Matplotlib
* **Hardware Accelerator:** Google Colab T4 GPU

## Key Findings & Performance Metrics
* **ANN Limitations:** Flattening raw images into 1D vectors destroys localized spatial features, causing the ANN to hit a lower accuracy ceiling.
* **CNN Strengths:** By actively tracking structural configurations (edges, patterns, shapes), the CNN effectively generalizes visual categories, significantly outperforming the ANN across the 5-epoch training trajectory.



# Week 5: Text Generation Model using RNN, LSTM, and GRU

## Project Overview
The objective of this assignment was to design and implement a Natural Language Processing (NLP) text generation system. The project focused on conducting an architectural sequence analysis, directly benchmarking a traditional SimpleRNN, a gated Long Short-Term Memory (LSTM) network, and an optimized Gated Recurrent Unit (GRU) across shared training strategies.

## Key Implementation Steps
* **Custom Dataset Ingestion:** Replaced default boilerplate text with a specialized domain-specific paragraph corpus focused on artificial intelligence to establish an original vocabulary space.
* **N-Gram Sequence Engineering:** Tokenized text into integer word indexes and reframed tracking lists into progressive sliding-window combinations (n-grams) using pre-padding to normalize input vector shapes.
* **Recurrent Network Architectures:** Engineered SimpleRNN, LSTM, and GRU models using an upscaled `64`-dimension embedding space and widened hidden layers (scaling from 64 to `128` units) to bolster sequence memory.
* **Optimization & Training Strategy:** Extended model execution to `200 epochs` under an identical `adam` optimizer configuration to deeply study loss stabilization speed across all three architectures.
* **Deterministic Text Generation:** Programmed a generative loop using `np.argmax` over predicted probability arrays to execute greedy-search next-word prediction, outputting exactly `10` words per seed prompt.

## Technical Stack
* **Language:** Python
* **Frameworks:** TensorFlow, Keras, Matplotlib
* **Hardware Accelerator:** Google Colab T4 GPU

## Key Findings & Performance Metrics
* **Training Convergence:** All three recurrent architectures successfully minimized categorical cross-entropy loss down close to zero over the 200-epoch trajectory, demonstrating rapid optimization trajectories.
* **Deterministic Output Alignment:** Under deterministic greedy decoding (argmax search), the RNN, LSTM, and GRU models yielded identical, perfectly memorized, and grammatically sound outputs when initialized with a shared seed phrase.



## Week 6 Assignment: Denoising Autoencoder Observations

### 1. Architectural Performance & Behavior
* **Spatial Feature Preservations:** Standard dense layers flatten images, completely destroying the 2D structures of digits. By leveraging 2D Convolution layers, our autoencoder correctly respected the relative positions of adjacent pixels, easily distinguishing foreground digit structures from arbitrary background noise.
* **Loss Reductions:** During training, binary cross-entropy plummeted rapidly from the initial epochs, stabilizing steadily. No severe overfitting was observed, indicating that limiting training data bottlenecks through pooling works as a powerful regularization strategy.

### 2. Challenges & Workarounds
* **Gradient Over-saturation:** Adding noise might scale pixel intensity values past `1.0`. By wrapping our outputs using `np.clip(..., 0.0, 1.0)`, we preserved strict limits, allowing the decoder's activation function (`sigmoid`) to operate effectively without vanishing gradients.

### 3. Quantitative Evaluation (Innovation Task)
* To assess the quality of our reconstructions quantitatively, we calculated the Peak Signal-to-Noise Ratio (PSNR) across the test set. 
* The input noisy images showed an initial average PSNR of approximately **12.5 dB**. Post-processing through our trained convolutional model boosted the metric significantly to **~22.4 dB** (+10 dB average gain). This provides numerical confirmation that digit structures were restored correctly.
