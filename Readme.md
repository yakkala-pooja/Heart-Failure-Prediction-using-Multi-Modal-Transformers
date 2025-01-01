## 1. Log in to Your Server

Log in to your remote server where the notebook files will be executed.

```bash
ssh <username>@<server-address>
```

## 2. Create a New Directory and Move Files

### 2.1 Transfer Files to the Server

Use `scp` to transfer files securely:

```bash
scp /path/to/Term_Project_Yakkala.zip <username>@<server-address>:/path/to/destination
```

### 2.2 Unzip the Files

After transferring, unzip the contents:

```bash
unzip Term_Project_Yakkala.zip
```

The directory will have the following structure:

```
├── Main.ipynb                         # Notebook for main model development
├── Dataset + Baseline.ipynb           # Notebook for baseline model setup
├── requirements.txt                   # Dependencies list
├── tabular_data.csv                   # Tabular data
├── time_series_data.npy               # Time-series data
└── bestmodel.pth                      # Best Model
```

## 3. Set Up Your Environment

### 3.1 Activate Your Conda Environment

Ensure Conda is installed. Activate the environment where dependencies will be installed:

```bash
conda activate <environment_name>
```

### 3.2 Install Dependencies

Run the following command to install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 4. Running the `Main.ipynb` Notebook

### 4.1 Start the Jupyter Notebook Server

Start Jupyter Notebook on the remote server:

```bash
jupyter notebook --no-browser --port=8880
```

### 4.2 Port Forwarding to Your Local Machine

On a second terminal on your **local machine**, forward the remote port to your local port:

```bash
ssh -N -f -L localhost:8880:localhost:8880 <username>@<server-address>
```

### 4.3 Access the Notebook and Run

Open a browser and go to:
[http://localhost:8880](http://localhost:8880)

### 4.4 `Main.ipynb` Workflow

#### 1. Loading Dataset
- Imports necessary libraries.
- Loads two datasets:
  - **Time-series data**: `time_series_data.npy`.
  - **Tabular data**: `tabular_data.csv`.
- Prints shapes and initial entries for verification.

#### 2. Preprocessing Dataset
- Encodes categorical features using `LabelEncoder`.
- Normalizes numerical columns using `StandardScaler`.

#### 3. Balancing and Train-Test Split
- Applies **SMOTEENN** to handle imbalanced datasets.
- Combines time-series and tabular data.
- Splits the combined data into **training** and **testing** sets.
- Normalizes tabular and time-series data independently.
- Converts data into PyTorch tensors and prepares `DataLoader` for training and evaluation.

#### 4. Model Training
- Defines a **Multimodal Transformer (MMTransformer)** architecture:
  - Processes both **tabular data** and **time-series data**.
  - Implements positional encoding for time-series features.
  - Fuses embeddings from the tabular and time-series modules in a **fusion layer**.
  - Performs classification based on the fused embeddings.
- Trains the model over **10 epochs**, saving the **best model** based on validation accuracy.

#### 5. Attention Mechanisms and Interpretability
- Applies post-hoc attention to analyze the importance of time-series and tabular embeddings.
- Visualizes attention weights as **heatmaps** for selected test samples.

#### 6. SHAP Analysis
- Uses **SHAP (SHapley Additive exPlanations)** to compute feature importance scores for the **tabular data**.
- Generates **SHAP summary plots** to visualize feature contributions to model predictions.

#### 7. LIME Explanations
- Uses **LIME (Local Interpretable Model-agnostic Explanations)** to analyze predictions on tabular data.
- Displays explanations for predictions in batches.

#### 8. Outputs
- Trained model is saved as `bestmodel.pth`.
- Visualizations of interpretability (SHAP plots, LIME results, attention heatmaps) are displayed for analysis.

---

## 5. Running `Dataset + Baseline.ipynb` Notebook

### 5.1 Start Jupyter Notebook for Baseline

Repeat **Step 4.1 to 4.3** but use a different port:

```bash
jupyter notebook --no-browser --port=8881
```

Port forwarding:

```bash
ssh -N -f -L localhost:8881:localhost:8881 <username>@<server-address>
```

Access:
[http://localhost:8881](http://localhost:8881)

### 5.2 Workflow of `Dataset + Baseline.ipynb`

#### Step 1: Data Generation

##### **Tabular Data**
- **Features** include:
  - Age
  - Sex
  - RestingBP
  - Cholesterol
  - MaxHR
  - And other critical attributes.
- A probabilistic model computes the likelihood of **HeartDisease** based on key features.

##### **Time-Series Data**
- Simulates signals like:
  - ECG
  - Heart Rate (HR)
  - Respiration Rate (RR)
- **Irregularities** are introduced for patients diagnosed with heart disease.

##### Data Saving
- **Tabular data** is saved as `tabular_data.csv`.
- **Time-series data** is saved as `time_series_data.npy`.

#### Step 2: Baseline Model Setup

##### Loading Data
- The tabular dataset (`tabular_data.csv`) is loaded.

##### Preprocessing
- **Categorical columns** are encoded using **Label Encoding**.
- Features are standardized with **StandardScaler**.

##### Train-Test Split
- The dataset is split into **training** and **testing** sets with an **80/20 split**.

##### Baseline Model
- A **Logistic Regression** model is initialized with the parameter:
  ```python
  class_weight='balanced'
  ```

#### Step 3: Model Training and Evaluation

##### Training
- The **Logistic Regression** model is trained on the training dataset.

##### Evaluation
- Predictions are made on the test dataset.
- Performance metrics are calculated:
  - Precision
  - Recall
  - F1-Score

#### Step 4: Results
- The **classification report** displays the baseline model's performance.
- This serves as a benchmark for comparison against advanced models.
