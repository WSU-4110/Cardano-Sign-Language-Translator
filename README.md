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

![278060231_570481434654106_4645321559362483949_n](https://user-images.githubusercontent.com/65133652/165195948-53ea474b-1c68-4b05-8172-8842b78db7da.png)
<p align="center">
  <img width="460" height="300" src = "https://user-images.githubusercontent.com/65133652/165196003-55ed7924-ffa4-4e3c-bdfa-7f6bbed223e9.png">
</p>

![274777769_677337986870020_7761917556052542863_n](https://user-images.githubusercontent.com/65133652/165196020-f792eaf7-2dac-4d13-8eb9-0936fc4e6f8c.png)

## **Video showing demo of the application**:

https://user-images.githubusercontent.com/65133652/165197453-eeb938f7-e89a-471d-a7e1-eaba8edc5a60.mp4

## Dataset:
Sign Language [MNIST](https://www.kaggle.com/datamunge/sign-language-mnist). Each letter is represented by an index 0-25 which corresponds to the letter in the alphabet A-Z


## Contributors:
- Ayon Chakroborty 
  - AI model development and training, optimization using GPU and CPU, connected database with Streamlit
- Luka Cvetko
  - OpenCv integration, detection algorithm, word_suggestion algorithm, connected Streamlit UI + AI model + OpenCv algorithm
- Patana Phongphila
  - UI design, database

