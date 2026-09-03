# QuCFTnet Methodology
1. Data Preprocessing

The CIC-IDS2018 traffic-flow data are first cleaned by removing non-numeric features, zero-variance features, infinite values, and unusable missing-value columns. The dataset is divided into training, validation, and testing subsets using a stratified 70:15:15 split. Min–Max normalization is fitted only on the training data and then applied to the validation and test sets to prevent data leakage.

2. Quantum-Inspired Feature Extraction

The normalized traffic features are processed using the proposed quantum-inspired feature extraction module. Trainable feature projection, amplitude encoding, Hadamard-inspired transformation, Pauli-inspired rotations, and weighted feature interactions are applied to generate an enhanced feature representation. This stage improves the representation of complex relationships within network traffic data.

3. Taylor-Series Nonlinear Transformation

The extracted quantum-inspired features are further transformed using a third-order Taylor-series expansion. This operation introduces nonlinear feature representations and enables the model to capture complex variations in network traffic. Cross-feature interactions are also incorporated to strengthen the representation of related traffic characteristics.

4. CNN-Based Hierarchical Feature Learning

The transformed features are subsequently processed using the CNN module. Conv1D layers are employed to learn local feature relationships, while batch normalization and pooling improve feature abstraction and reduce redundant information. Dropout is used to improve generalization, followed by fully connected layers for high-level feature representation.

5. Intrusion Classification

The learned high-level representation is passed to the final Softmax classifier for intrusion detection. For binary classification, the classifier predicts the probability of each traffic sample belonging to the Normal or Attack class. The class with the highest predicted probability is selected as the final classification result.