# SIGN LANGUAGE DATASET
The Sign Language MNIST is presented here and follows the CSV format with labels and pixel values in single rows.

The American Sign Language letter database of hand gestures represent a multi-class problem with 24 classes of letters (excluding J and Z which require motion).
Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions).

The training data (27,455 cases) and test data (7172 cases) are approximately half the size of the standard MNIST but otherwise similar with a header row of label, pixel1,pixel2….pixel784 which represent a single 28x28 pixel image with grayscale values between 0-255. 

### Combined these pixels will create an image representing one of the ASL alphabet letters:
![amer_sign2](https://user-images.githubusercontent.com/76793940/157327993-9d894b22-dfa0-4ccf-8f85-b3a989de74b4.png)

