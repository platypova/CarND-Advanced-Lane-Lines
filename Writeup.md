## Project writeup 
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Camera Calibration

The code for this step is contained in the second code cell of the IPython notebook 'Advanced_lane_finding.ipynb' below the heading "Camera Calibration". 

This step uses images with chessboard cards located in ./camera_cal directory. Firstly I choose number of inside corners in x and y - 9 and 6. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result (here are distorted and undistorted images):

[image1]: ./output_images/chessboard_distorted.jpg "Distorted"
[image2]: ./output_images/chessboard_undist.jpg "Undistorted"

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

[image3]: ./test_images/straight_lines1.jpg "Test image"

I used distortion coefficients and matrix calculated in 4th code cell of notebook 'Advanced_lane_finding.ipynb'. In orer to get undistorted image I used cv2.undistort() function. the result was:

[image4]: ./output_images/straight_lines1_undist.jpg "Test image undistorted"


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.  Code for this step is located in code cells below the heading "Gradients and color transforms" in the IPython notebook 'Advanced_lane_finding.ipynb'. 
I wrote several functions to  apply different types of thresholding in order to obgtain binary image with lanes:
- function 'abs_sobel_thres' where x or y gradient is applied with the OpenCV Sobel() function and the absolute value is taken;
- function 'mag_thresh' where both x and y gradient is applied with the OpenCV Sobel() function and the gradient magnitude is calculated;
- function 'dir_threshold' that applies Sobel x and y,then computes the direction of the gradient;
- function 'hls_threshold' that applies thresholding for h and s channels of the image.
Each function creates a binary mask where thresholds are met.
I tried combining different functions and thershold values. I created function 'edges_pipeline' using all these functions in the best combination I managed to found. It allowed me to obtain images where pixels belonging to lane markers are well-observed. Examples are here: 

[image5]: ./test_images/straight_lines1.jpg "Test image"
[image6]: ./output_images/binary_lanes.png "Test image binarized"


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in code cells below the heading "Region of interest & Perspective transform" of the IPython notebook 'Advanced_lane_finding.ipynb'.  
The `perspective_transform()` function takes as inputs an image (`img`), while source (`src`) and destination (`dst`) points are defined inside the function. I chose the hardcode the source and destination points in the following manner:
```python
 height = img.shape[0]
    width = img.shape[1]
    offset = 100
    # Points in the source image:
    src_points = np.array([[(width // 2 - 76, height * 0.625), 
                          (width // 2 + 76, height * 0.625), (offset,height),(width,height)]], dtype=np.float32)
    # Points in the destination image:
    dst_points = np.array([[(offset, 0), 
                          (width-offset, 0), (offset,height), (width-offset,height)]], dtype=np.float32)
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 564, 450      | 100, 0        | 
| 716, 450      | 1180, 0       |
| 100, 720      | 100, 720      |
| 1280, 720     | 1180, 0       |

The function reutrns warped image and perspective transform matrix. As the result, we obtain binary bird-eye view image of our region of interest, which we use to find pixels belonging to lane markers. Here is the example of this function working:

[image7]: ./output_images/binary_birdeye.png "Bird-eye view binarized test image"

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
Code for the next step is located in part called "Histogram peaks & sliding window" in the IPython notebook 'Advanced_lane_finding.ipynb'. 
Here I use 2 functions: 'find_lane_pixels' and 'fit_polynomial'.
'find_lane_pixels' function does the following:
- takes a histogram of the bottom half of the image
- finds the peak of the left and right halves of the histogram. These will be the starting point for the left and right lines
- the number of sliding windows is chosen, their size is chosen
- the x and y positions of all nonzero pixels in the image are identified
- then for each window the nonzero pixels in x and y within the window are identified, and appended to a list.
'fit_polynomial' function fits positions of the detected pixels to a polinomial.

Code for the next step is located in the part below the heading 'Skip the sliding windows'.Polynomial fit values are taken from the previous frame, if lane was detected. It allows to start search around previous results and help to spend less resources.

[image8]: ./output_images/lane_windows.png "Test image binarized"

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in code cells below the heading "Computing radius of curvature" in the IPython notebook 'Advanced_lane_finding.ipynb'. 
I wrote function 'measure_curvature_real' which calculates radius and center offset.
Firstly, conversions in x and y from pixels space to meters are defined. Radius is calculated basing on those conversions, coefficients obtained while fitting lane marker pixels to polinomial and maximum y-values of right and left lanes. Center offset for the vehicle is calculated as difference between image center and lane center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code cells below the heading "Inverse Transform" in the IPython notebook 'Advanced_lane_finding.ipynb' in the function 'inverse_transform'. Here is an example of my result on a test image:

[image9]: ./output_images/straight_lines1_lane_drawn.jpg "Test image with lane plotted back down onto the road"

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./video_lanes_detected.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The approach I learned and used here seems to work much better than approach used in previous project.

I think that one of the drawbacks is the part where we obtain binary image using heuristically defined thersholds. It's obious that we've tested the technique for very good conditions: on the video the weather is sunny, we don't see any water on the road surface, etc. I suppose some problems may occur in other conditions with other levels of lightness, for example. And the thresholds don't seem to be universal. But the plus is that we find lane marker pixels basing on different information (HLS model, sobel operator etc). We cant tune it and improve the results quite easily.

I think that it could be an idea to have some external information about weather, day time in order to have a set of thresholds fitting well several combintaions of such external information. I think it may help us to do the pipline more stable.

