\# 🩺 Breast Cancer Classification using Deep Learning



!\[Python](https://img.shields.io/badge/python-3.11-blue)

!\[TensorFlow](https://img.shields.io/badge/tensorflow-2.13-orange)

!\[License](https://img.shields.io/badge/license-MIT-green)



This project focuses on detecting \*\*breast cancer from histopathology images\*\* using \*\*Deep Learning\*\*.  

It implements multiple approaches including \*\*Baseline CNN\*\*, \*\*Class Weighting\*\*, \*\*Data Augmentation\*\*, and \*\*Transfer Learning with MobileNetV2\*\*.



The main goal is to \*\*maximize cancer detection (recall)\*\*, which is crucial for medical diagnosis.



---



\## 📌 Problem Statement



Breast cancer is one of the leading causes of death among women worldwide.  

Early detection significantly improves survival rates.



This project builds a \*\*binary image classifier\*\* to predict whether a tissue image is:



\- \*\*Benign (0)\*\*

\- \*\*Malignant (1)\*\*



\## 📂 Dataset



\- \*\*Dataset Name:\*\* Breast Cancer Histopathological Images (IDC)  

\- \*\*Image Size:\*\* 50×50 pixels (RGB)  

\- \*\*Total Images:\*\* ~267,000+  

\- \*\*Classes:\*\* Benign, Malignant  

\- \*\*Source:\*\* Public medical dataset  

---



\## 📁 Project Structure

Breast\_Cancer\_Classification/

│

├── data/ # Raw dataset (not tracked in GitHub)

│ ├── benign/

│ └── malignant/

│

├── models/ # Saved trained models

│ ├── baseline\_cnn.h5

│ ├── class\_weighted\_model.h5

│ └── mobilenet\_finetuned.h5

│

├── results/ # Plots \& metrics

│ ├── accuracy\_plot.png

│ └── confusion\_matrix.png

│

├── src/ # Source code

│ └── cancernet.py

│

├── README.md

├── requirements.txt

└── .gitignore



---



\## 🧠 Models Implemented



\### 1️⃣ Baseline CNN

\- Custom CNN architecture

\- Binary cross-entropy loss

\- Adam optimizer



\### 2️⃣ Class Weighting

\- Handles severe class imbalance

\- Improves recall for malignant class



\### 3️⃣ Data Augmentation

\- Rotation, zoom, horizontal flip

\- Improves generalization



\### 4️⃣ Transfer Learning (MobileNetV2)

\- Pretrained on ImageNet

\- Frozen base layers

\- Custom classification head



\### 5️⃣ Fine-Tuning

\- Unfreezes top layers of MobileNetV2

\- Lowers learning rate

\- Better feature adaptation



---



\## 📊 Results



\### Accuracy Plot

!\[Accuracy Plot](results/accuracy\_plot.png)



\### Confusion Matrix

!\[Confusion Matrix](results/confusion\_matrix.png)



\### Sample Metrics (Example from final run)

| Class      | Precision | Recall | F1-score | Support |

|----------- |---------- |------- |--------- |-------- |

| Benign (0) | 0.72      | 0.54   | 0.62     | 38234   |

| Malignant (1)| 0.29    | 0.47   | 0.36     | 15171   |

| \*\*Accuracy\*\*|          |        | 0.52     | 53405   |



---



\## ⚙️ Technologies Used



\- Python 3.11  

\- TensorFlow / Keras  

\- NumPy  

\- Scikit-learn  

\- Matplotlib  

\- OpenCV  



---



\## 🚀 How to Run



1\. \*\*Clone the repo:\*\*

```bash

git clone https://github.com/J-Nandini-eng/Breast\_Cancer\_Classification.git

cd Breast\_Cancer\_Classification



Create a virtual environment \& install dependencies:



python -m venv venv

\# Windows

venv\\Scripts\\activate

\# Linux / Mac

source venv/bin/activate



pip install -r requirements.txt





Run the main script:



python src/cancernet.py





The script will train the model, compute class weights, optionally apply data augmentation, perform transfer learning, and save the model and results in models/ and results/.



\## 📌 License

MIT License



