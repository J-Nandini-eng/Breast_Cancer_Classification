\# 🩺 Breast Cancer Classification using Deep Learning



This project focuses on detecting \*\*breast cancer from histopathology images\*\* using \*\*Deep Learning techniques\*\*.  

Multiple approaches were implemented and compared, including a \*\*Baseline CNN\*\*, \*\*Class Weighting\*\*, \*\*Data Augmentation\*\*, and \*\*Transfer Learning with MobileNetV2\*\*.



The goal of the project is to \*\*maximize cancer detection (recall)\*\* rather than just accuracy, which is critical in medical diagnosis tasks.



---



\## 📌 Problem Statement



Breast cancer is one of the leading causes of death among women worldwide.  

Early detection significantly improves survival rates.



This project builds a \*\*binary image classifier\*\* to predict whether a tissue image is:

\- \*\*Benign (0)\*\*

\- \*\*Malignant (1)\*\*



---



\## 📂 Dataset



\- \*\*Dataset Name:\*\* Breast Cancer Histopathological Images (IDC)

\- \*\*Image Size:\*\* 50×50 pixels (RGB)

\- \*\*Total Images:\*\* ~267,000+

\- \*\*Classes:\*\*

&nbsp; - Benign

&nbsp; - Malignant

\- \*\*Data Source:\*\* Public medical dataset (IDC)



---



\## 📁 Project Structure



Breast\_Cancer\_Classification/

│

├── data/

│ ├── benign/

│ └── malignant/

│

├── models/

│ ├── baseline\_cnn.h5

│ ├── class\_weighted\_model.h5

│ └── mobilenet\_finetuned.h5

│

├── results/

│ ├── confusion\_matrix.png

│ ├── accuracy\_plot.png

│

├── src/

│ └── cancernet.py

│

├── README.md

└── requirements.txt





---



\## 🧠 Models Implemented



\### 1️⃣ Baseline CNN

\- Custom CNN architecture

\- Binary cross-entropy loss

\- Adam optimizer



\### 2️⃣ Class Weighting

\- Handled severe class imbalance

\- Improved malignant class recall



\### 3️⃣ Data Augmentation

\- Rotation

\- Zoom

\- Horizontal flip

\- Improved generalization



\### 4️⃣ Transfer Learning (MobileNetV2)

\- Pretrained on ImageNet

\- Frozen base layers

\- Custom classification head



\### 5️⃣ Fine-Tuning

\- Unfroze top layers of MobileNetV2

\- Lower learning rate

\- Improved feature adaptation



---



\## ⚙️ Technologies Used



\- Python 3.11

\- TensorFlow / Keras

\- NumPy

\- Scikit-learn

\- Matplotlib

\- OpenCV



---

&nbsp;   \*\*Results section\*\* 



\## 🚀 How to Run the Project



```bash

python src/cancernet.py


\## 📊 Model Performance Summary



\- Training Accuracy: ~85%

\- Validation Accuracy: ~76%

\- Malignant Recall: ~47–51%

\- Dataset was highly imbalanced, so recall was prioritized over accuracy



The model is intended for \*\*educational and research purposes only\*\*, not clinical use.



---



\## 👩‍💻 Author



\*\*Nandini\*\*  

GitHub: https://github.com/J-Nandini-eng









