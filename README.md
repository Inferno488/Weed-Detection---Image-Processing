# Weed-Detection---Image-Processing
This is a self-designed algorithm that detects the presence of weed in agricultural farms through fundamentals of image processing.


To use this pre-requisties are OpenCV & NumPY.

This weed detection algorithm works on the fundamentals of image processing.

The steps it follows are:
1. Image acquisition
2. Conversion to HSV & Resize
3. Eroding, Dilating & Gaussian Blurring the image with a 7,7 Kernel.
4. Masking the resultant image with a specific range of HSV values.
5. Finally tracing countours using inbuilt opencv method of findContours.
