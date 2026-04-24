# Data Directory

Raw data is **not tracked in this repository**.

## Download Instructions

```bash
# Requires Kaggle CLI configured with your API key (~/.kaggle/kaggle.json)
kaggle datasets download -d ajithdari/multi-modal-breast-cancer-dataset
unzip multi-modal-breast-cancer-dataset.zip -d raw/
```

## Expected Structure

```
data/raw/
├── benign/
│   ├── images/     # ultrasound images (.png)
│   └── masks/      # segmentation masks (.png)
├── malignant/
│   ├── images/
│   └── masks/
└── normal/
    ├── images/
    └── masks/
```
