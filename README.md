# AP2
# U-net-Breast-Cancer-Ultrasound-ImageSegmenatation

------
U-Net Breast Ultrasound Image Segmentation

This project focuses on breast ultrasound image segmentation, leveraging U-Net and U-Net++ architectures for effective delineation of tumor regions. The goal is to assist in automated and accurate identification of breast cancer from ultrasound images, improving diagnostic support and advancing medical imaging applications.

Overview

	•	Model Architectures: U-Net, U-Net++
	•	Task: Semantic segmentation of breast ultrasound images to isolate and highlight tumor regions.
	•	Dataset: Breast ultrasound images (details of the dataset source, if publicly available, or custom dataset description).
	•	Results: Segmentation accuracy and comparison metrics between U-Net and U-Net++ models.

Project Structure

	•	data/: Contains the dataset and any pre-processing scripts.
	•	models/: Includes implementations of U-Net and U-Net++ architectures.
	•	training/: Code for training the models with options for tuning parameters.
	•	evaluation/: Scripts for evaluating model performance, including accuracy, dice coefficient, and IOU.
	•	notebooks/: Jupyter notebooks with visualizations and detailed analysis.
	•	README.md: Project documentation (this file).

Installation

  1.	Clone this repository: git clone https://github.com/woogunny/U-net-Breast-Cancer-Ultrasound-ImageSegmenatation.git
                            cd U-netBreastUltrasoundSegmentation
  2.	Install the required packages: pip install -r requirements.txt

Usage

Training

To train the U-Net model: python model/unet.py --epochs 120 --batch-size 8 --learning-rate 0.0001
To train the U-Net++ model: python model/unetpp.py --epochs 120 --batch-size 8 --learning-rate 0.0001

Evaluation

Evaluate model performance on the test dataset: python evaluation/evaluate.py --model unet
                                                python evaluation/evaluate.py --model unet_plus_plus

Results

Comparative metrics (Dice Coefficient, IOU, etc.) for U-Net and U-Net++ models:

Visualization

Example segmentation results: 
![prediction_testimage_unet1](https://github.com/user-attachments/assets/b511f43a-620c-479a-9079-35b819886f10)

![prediction_testimage_unet++_1](https://github.com/user-attachments/assets/94df487a-401c-41ab-964a-463bc1b585d0)

License

This project is licensed under the KAICT License.

Let me know if you’d like any customization on this!

