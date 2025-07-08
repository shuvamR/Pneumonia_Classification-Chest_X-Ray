# Hyperparameters Used
In this pneumonia classification task using a pre-trained InceptionV3 model, the following hyperparameters were carefully selected to balance performance, generalization, and computational efficiency:

| **Hyperparameter**            | **Value**                                                  | **Why This Value Was Chosen**                                     | **Effect on Model**                                             |
| ----------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------- |
| **Image Size**                | `75 × 75` pixels                                           | Reduces computation while preserving important X-ray details      | Efficient training, lower memory usage, good feature retention  |
| **Batch Size**                | `32`                                                       | Common standard; balances training stability and memory usage     | Smooth gradient updates; works well on GPU                      |
| **Epochs (Transfer)**         | `50`                                                       | Enough to train only the top layers after freezing the base model | Allows learning from scratch without overfitting                |
| **Epochs (Fine-tune)**        | `100`                                                      | More training needed when unfreezing deeper base layers           | Gradually adapts pretrained features to pneumonia detection     |
| **Learning Rate (Transfer)**  | `1e-4`                                                     | Fast learning for newly added dense layers                        | Speeds up convergence during initial training                   |
| **Learning Rate (Fine-tune)** | `1e-5`                                                     | Lower value to update pretrained weights carefully                | Prevents large changes, preserves useful learned features       |
| **Dropout Rate**              | `0.5`                                                      | Standard value to prevent co-adaptation of neurons                | Reduces overfitting by forcing robustness                       |
| **Class Weights**             | Calculated using `compute_class_weight()`                  | Balances "Normal" and "Pneumonia" labels in an imbalanced dataset | Ensures minority class (Pneumonia) is given proper importance   |
| **Data Augmentation**         | Horizontal flip, rotation (5%), zoom (10%), contrast (10%) | Simulates variations in real-world X-rays                         | Improves generalization, increases effective training data size |
