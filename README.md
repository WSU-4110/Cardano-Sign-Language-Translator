# Cardano-Sign-Language-Translator

## Description:
***Cardano*** is a program which recognizes hand gestures from a camera source. The main use of Cardano is the conversion of American Sign Language(ASL) to any other spoken language.

**Cardano offers real time translation with Roaming capabilities:**
- No excess equipment
- Translate active Sign Language
- Non-Stationary

## Technology:
By combining the Pytorch Deep Learning module and [OpenCV](https://docs.opencv.org/4.5.5/) a image processing library to create a deep neural network model for pose recognition.

Translation to mutliple languages done using the [Google Translation API](https://cloud.google.com/translate/). 

Hosted on the Streamlit framework.
## Context:
- To help the deaf or mute
- Translate sign language
- Remove the communication barrier
- Less than 1% of US population know sign language

## Design - Screenshots:

//add pics


## Dataset
Sign Language [MNIST](https://www.kaggle.com/datamunge/sign-language-mnist). Each letter is represented by an index 0-25 which corresponds to the letter in the alphabet A-Z


## Contributors:
- Ayon Chakroborty 
  - AI model development and training, optimization using GPU and CPU, connected database with Streamlit
- Luka Cvetko
  - OpenCv integration, detection agorithm, word_suggestion algorithm, connected Streamlit UI + AI model + OpenCv algorithm
- Patana Phongphila
  - UI design, database

