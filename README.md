# 🧠 Brain Tumor Classification using ResNet50

This project leverages **transfer learning** with a modified ResNet50 model to classify brain tumors from MRI scans into three categories:

- **Meningioma**
- **Glioma**
- **Pituitary Tumor**

Built with TensorFlow and trained using **5-fold cross-validation**, the model supports single-file classification via an interactive UI and is based on the [Kaggle BraTS 2015 dataset](https://www.kaggle.com/datasets/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015/data).

---

## 🗂️ Project Structure

```
brain-tumor-classification/
├── brain_tumor_classifier_resnet50.ipynb   # Cleaned notebook
├── best_model_fold_1.keras                 # Best trained model (via Git LFS)
├── requirements.txt                        # Required libraries
├── README.md                               # This file
├── .gitignore                              # Ignores large data files
├── .gitattributes                          # Tracks model files via Git LFS
└── data/                                   # Not included – local only
```

---

## 📦 Dataset

Due to size limitations, the dataset is **not included** in this repo.  
Please download it from:

🔗 [Brain Tumor Segmentation in MRI (BraTS 2015) on Kaggle](https://www.kaggle.com/datasets/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015/data)

Once downloaded, place all `.mat` files into a folder like:

```
/data/
  ├── image_001.mat
  ├── image_002.mat
  └── cvind.mat
```

Update these paths in the notebook:

```python
path_to_mat_files = 'data/'
cvind_path = 'data/cvind.mat'
```

---

## 🚀 How It Works

### 🧠 Model Architecture

- Base: **ResNet50** (pretrained on ImageNet)
- Additional CNN layers for feature enhancement
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy

### 🔁 Training Strategy

- **5-fold cross-validation**
- Data augmentation (rotation, shift, zoom, flip)
- EarlyStopping + ModelCheckpoint
- Best model for each fold saved as `.keras`

---

## 🔍 Single File Classification

The notebook includes a UI (via `ipywidgets`) to classify a single `.mat` file:

1. Provide the path to a saved model (`.keras`)
2. Provide the path to a `.mat` MRI file
3. The UI will display the predicted tumor type and confidence score.

---

## 📊 Evaluation

For each fold, the notebook displays:
- Accuracy & loss plots
- Fold-specific metrics
- Final result summary:
```python
fold_results = {
  1: {'val_loss': 0.22, 'val_acc': 0.94},
  ...
}
```

---

## 📦 Setup Instructions

### 🖥️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 💡 Possible Enhancements

- Add Grad-CAM visualizations
- Web UI using Streamlit or Gradio
- Ensemble predictions from multiple folds

---

## 🛡️ License

This project is intended for academic and educational use.  
Please credit if reused.

---

## 🙏 Acknowledgments

- Dataset: [Kaggle – BraTS 2015](https://www.kaggle.com/datasets/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015/data)
- ResNet50 architecture: [Keras Applications](https://keras.io/api/applications/)
