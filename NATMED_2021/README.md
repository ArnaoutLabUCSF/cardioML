## If you use code or data from this repository, please cite:
Rima Arnaout, Lara Curran, Yili Zhao, Jami C. Levine, Erin Chinn & Anita J. Moon-Grady. An ensemble of neural networks provides expert-level prenatal detection of complex congenital heart disease. Nature Medicine volume 27, pages 882–891 (2021)(PMID 33990806)

The code here is covered under the GNU General Public License.

## Installation & Requirements

Install requirements from requirements.txt
	Can use standard pip or conda installs (e.g. "pip install opencv-python"; <10 minutes)


## Instructions

To convert a 300x400 png to a cropped and downsampled 80x80 png:
	Update the input filename, cropped dimensions, and scale in process_image.py
	Command line run: python process_image.py

To train the 6-view classification model:
	Update the train file and finished model path in train_classifier_model.py
	Command line run: python train_classifier_model.py

To train a binary classification model:
	Update the train file and finished model path in train_classifier_model.py
	Update label_map and change loss function to binary_crossentropy
	Command line run: python train_classifier_model.py

To train the 4 chamber segmentation model:
	Update the train files and finished model path in train_4chamber_model.py
	Command line run: python train_4chamber_model.py