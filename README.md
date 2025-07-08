# Pneumonia Classification of Chest X-Ray by fine tuning InceptionV3 Model

***Dataset Used: [PneumoniaMNIST](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist)***

## ***About Dataset***

### **PneumoniaMNIST**
A curated collection of chest X-ray images for binary classification of pneumonia:
#### Images
- Grayscale, 28×28 pixels
- Stored in NumPy arrays: ```train_images```, ```val_images```, ```test_images```
#### Labels
- Binary: 1 = pneumonia present, 0 = normal
- Stored alongside ```images astrain_labels```, ```val_labels```, ```test_labels```
#### Dataset splits
- Train: ```3 883 images```
- Validation: ```524 images```
- Test: ```624 images```

***[Hyperparameters Used](https://github.com/shuvamR/Pneumonia_Classification-Chest_X-Ray/blob/main/note.md)***
## Steps to Run

### 1. FOR CLOUD BASED RUNTIME ENVIRONMENT LIKE KAGGLE
* Step 1: Download the .iypnb notebook file.
* Step 2: Go to Kaggle. Click on "New" option and select import notebook.
* Step 3: Open the Notebook.
* Step 4: Navigate on the Right Panel and click on "Add Input"
* Step 5: Search for "[PneumoniaMNIST](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist)" and Add the dataset.
* Step 6: Execute the code
  
### 2. FOR LOCAL RUNTIME ENVIRONMENT LIKE JUPYTER OR SPYDER
*  Step 1: Download the .iypnb notebook file and [PneumoniaMNIST](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist)
*  Step 2: Open the Notebook.
*  Step 3: Install the [dependencies](https://github.com/shuvamR/Pneumonia_Classification-Chest_X-Ray/blob/main/requirements.txt) using the command ```pip install -r requirements.txt```
*  Step 4: Execute the code.

*Note: If you see any error related to file location, then update the file paths.*


## Evaluation Strategy 

***A. Choose 3-appropriate metrics and justify your choices.***
| Metric                     | What it means                                    | Why it is important for pneumonia detection                                            |
| -------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **AUC** (Area Under Curve) | How well the model can tell apart Normal and Pneumonia              | Helps check if model can **separate the two classes well**, even if data is imbalanced |
| **Recall** (Sensitivity)   | Out of all actual pneumonia cases, how many the model caught        | Important because **missing pneumonia is risky** for the patient                       |
| **Precision**              | Out of all cases the model says are pneumonia, how many are correct | Important to **avoid false alarms** and **unnecessary treatments**                     |

Also I included ```binary_accuracy``` to get a basic overall idea of how often the model gets predictions right (either class 0 or 1 i.e. Healthy or Pneumonia).But we focus more on AUC, Precision, and Recall, because they are more meaningful for imbalanced data.

***B. Discuss how you detect and mitigate class imbalance in the training set.***
* Class imbalance means one class has more samples than the other. This can make the model biased toward the majority class (Normal or Healthy) and ignore the minority class (Pneumonia). We can detect imbalance by counting how many images belong to each class.
* To mitigate this, we can use:
    1. **Class Weights**: Give more importance to the minority class during training so the model treats both classes fairly.
    2. **Data Augmentation**: Create more training examples by slightly modifying existing images (e.g., flipping, rotating, zooming), which helps improve model learning, especially for the minority class.
* These techniques help the model learn better and perform well on both classes.

***C. Describe measures taken to prevent over-fitting***
| Method                  | What It Does                            | Purpose (How It Prevents Overfitting)                           |
| ----------------------- | -------------------------------------------------------------- | --------------------------------------------------------------- |
| **Data Augmentation**   | Flips, rotates, zooms, and adjusts contrast of training images | Adds variety to training data and prevents memorizing           |
| **Dropout Layer (0.5)** | Turns off random neurons during training                       | Makes the model less dependent on specific features             |
| **Transfer Learning**   | Uses a pre-trained InceptionV3 with frozen layers initially    | Starts with strong, general features, avoids overfitting early  |
| **Fine-Tuning** | Only unfreezes the last 50 layers of the base model            | Updates higher-level features without disturbing base knowledge |



