# 🧠 Digit Classification using Classical ML Models and PCA

This project demonstrates the classification of handwritten digits (0–9) using classical machine learning models. The dataset used is the popular `load_digits` dataset from `scikit-learn`.

Two versions of the model are implemented:
- 🟡 A baseline model without dimensionality reduction.
- 🔵 A model that uses Principal Component Analysis (PCA) to reduce input dimensions before classification.

## 📁 Project Structure

- notebooks/digit_classification_baseline.ipynb: Baseline model without PCA.
- notebooks/digit_classification_with_pca.ipynb: Model using PCA for dimensionality reduction.
- README.md: Project documentation.
- requirements.txt: Dependencies list (optional).

## 🤖 Algorithms Used

The following machine learning algorithms are applied and compared:

- 🔷 K-Nearest Neighbors (KNN)
- 🔶 Support Vector Machines (SVM)
- 🧠 Artificial Neural Networks (ANN)
- 🌲 Random Forest Classifier

## 📊 Evaluation Metrics

To measure model performance, the following metrics are used:

- ✅ Accuracy
- 📉 Confusion Matrix
- 📌 (Optionally) Precision, Recall, and F1-Score

Bar charts are used to compare performance across algorithms visually.

## ⚙️ How to Run

1. Clone the repository.
2. Create and activate a virtual environment:

   Windows:
   python -m venv venv  
   venv\Scripts\activate

   macOS/Linux:
   python3 -m venv venv  
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Open and run the notebooks in Jupyter Notebook or VS Code.

## 🛠 Requirements

- 🟪 numpy
- 🟫 pandas
- 🟦 matplotlib
- 🟩 scikit-learn

If `requirements.txt` is not available, install manually:
pip install numpy pandas matplotlib scikit-learn

## 📚 Dataset

The project uses `load_digits` from `sklearn.datasets`, a dataset of 1,797 grayscale images (8x8 pixels) of handwritten digits.

## 💻 Developer

This project was developed by Mohammad Khademi as part of a machine learning portfolio for academic and professional development.

## ✅ License

This project is open-source and available for educational use. Contributions and feedback are welcome.
