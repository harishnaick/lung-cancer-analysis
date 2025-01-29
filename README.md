
# **Lung Cancer Detection using Deep Learning**

## **Overview**
This project aims to detect lung cancer from CT scan images using deep learning models. The model is trained on a dataset of lung cancer images, including **benign, malignant, and normal** cases. The system predicts whether an input image is cancerous or not.

## **Features**
- 🏥 **Deep Learning-Based Detection**: Uses **3D CNN, DCGAN, and GANN** models for accurate lung cancer classification.  
- 📊 **Data Visualization**: Generates **bar plots, line charts, and pie charts** to analyze the dataset.  
- 🌐 **Web Application**: Built using **Flask and Streamlit** for an interactive UI to upload and analyze images.  
- 🔍 **Multiple Model Support**: Uses `lungcancer.h5` and `lungcancerbig.h5` trained on **histopathological and CT scan images**.  
- 🖼 **Synthetic Image Generation**: Uses **DCGAN and AD-GAN** to create additional lung cancer images.  

## **Dataset**
- The dataset consists of **15,000 images** categorized into:
  - **Benign (Non-Cancerous)**
  - **Malignant (Cancerous)**
  - **Normal (Healthy Lungs)**
- The images are preprocessed and resized before feeding into the deep learning model.

## **Technologies Used**
- **Python** (TensorFlow, Keras, PyTorch, OpenCV, NumPy, Pandas, Matplotlib)
- **Machine Learning Models** (3D CNN, DCGAN, AD-GAN, GANN)
- **Flask & Streamlit** (Web UI for predictions)
- **Jupyter Notebook** (For model training and evaluation)

## **Installation & Setup**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/lung-cancer-detection.git
   cd lung-cancer-detection
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Flask web app**:
   ```bash
   python app.py
   ```
4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## **Usage**
- Upload a **CT scan image** through the web interface.
- The model processes the image and predicts if it is **benign, malignant, or normal**.
- View the **classification results** and **confidence scores**.

## **Results & Accuracy**
- The **3D CNN model** achieved **high accuracy** in classifying lung cancer.
- **DCGAN-generated synthetic images** helped improve model performance.
- **GANN-based model** further optimized classification results.

## **Screenshots**
📌 *Add UI screenshots here (upload images in the repository and link them in markdown).*

## **Future Enhancements**
- Improve accuracy using **transformer-based models**.
- Deploy the app on **AWS/GCP for cloud-based inference**.
- Implement **real-time image preprocessing**.

## **Contributors**
- **Your Name** – *Machine Learning Engineer*
- **Other Contributors (if any)**

## **License**
This project is licensed under the **MIT License**.

---🚀
