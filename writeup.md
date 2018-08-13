<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# **Advanced Lane Finding Project**

![Alt Text](https://media.giphy.com/media/4EFsn95JAPySiR03Yo/giphy.gif)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1a]: ./writeup_images/undistort_output.png 
[image1b]: ./writeup_images/test1_undistort.png
[image2]: ./writeup_images/test1_thresh.png 
[image3]: ./writeup_images/test1_mask.png 
[image4a]: ./writeup_images/test1_undistorted_and_warped.png
[image4b]: ./writeup_images/test1_transformed.png
[image5a]: ./writeup_images/test1_histogram.png
[image5b]: ./writeup_images/lane_detection.png
[image6a]: ./writeup_images/curvature.png
[image6b]: ./writeup_images/output.png
[video1]: ./project_video.mp4 "Video"

# [Rubric Points](https://review.udacity.com/#!/rubrics/571/view) 

### Here I will consider the above rubric points individually and describe how I addressed each point in my implementation.  

---


## Camera Calibration

All camera images are distorted due to the curvature of the lens. This affects the shape and sizes of objects in the image and thus inhibits our ability to accurately determine the position of our self-driving car, the direction of the road, and also the position of other objects in our field of view. It is therefore imperative that we first recover the ground-truth image before attempting any of these things. 

To do this, we will use the same camera from the self-driving car to take a series of photographs of chessboards in different positions. Chessboards are particularly useful since we know that they should have a precise two-dimensional lattice strucutre. We start by preparing "object points," which will be the (x, y, z) coordinates of the chessboard corners in the real world. We will assume the chessboard is fixed on the x-y plane at z=0, such that the object points are the same for each calibration image. 

We need about 20 different chessboard images to successfully calibrate the camera. For each image, the "object points" will be the same array of lattice points - we are essentially trying to map all of the camera images to the same lattice (the chessboard is the same in the real world in all 20 cases regardless of how the camera has distorted it). 

As we loop through all 20 images, we append a copy of these "object points" to the array `objpoints`. We then use the OpenCV function `cv2.findChessboardCorners` to detect the (x,y) pixel positions of corners in each chessboard image and append these to the array `imgpoints`. We then apply the OpenCV function `cv2.calibrateCamera()` to the arrays `objpoints` and `imgpoints` to compute the camera calibration matrix and distortion coefficients that successfully map the input camera image to the desired regular lattice structure in the output image.

Armed with the calibration matrix and distortion coefficients, we now have a mathematical understanding of the distortion caused by our particular camera lens. We are now able to input these distortion parameters to the OpenCV function `cv2.undistort()` to undistort *any image* taken using this camera.

Below we can see the distortion removal applied to one fo the chessboard calibration images: 

![alt text][image1a]


## Lane Detection Pipeline (single images)

Our task is to track the pane position and curvature on a .mp4 video file containing footage of a car driving along a road. Our approach will be to split the video file into individual images and feed these through the following pipeline one at a time:

#### 1. Remove lense distortion from the image.

Having obtained the camera calibration matrix and distortion coefficients above using the chessboard images, these parameters can be saved as pickle files so we can call upon them later.

In particular, the first step in our pipeline is to undistort all images so that we are performing lane detection with the ground-truth image e.g. 

![alt text][image1b]
Notice how the edges of the image get strectched. This is particularly obvious if you look at the position of the white car.

#### 2. Create a thresholded binary image

The next step is to apply a selection of thresholding procedures to the image so as to help identify the lane location. For me, this was the most important step of the pieline as smal changes led to drastically different results. After a lot of experimentation, I finally settled on the following combination of colour thresholds:

**Red Channel of RGB Colour Space (R):** 215 <= R <= 255

**L Channel of Luv Colour Space (L):** 185 <= L <= 255 

Whilst most people are familiar with the R channel, they are probably less aware of the L channel. The idea here is to use a three-dimensional mathematical transformation taking the conventional RGB colour space with a cubic geometry to a different geometry. Depending on the particular geometry produced, this new colour space will be sensitive to different things. In the case of hte Luv colour space, L measures luminosity whilst u and v measure chromaticity.

In the middle image below, we can see the effect of each of these thresholding techniques. The red thresholding of the original image is shown in red. We see that this is very robust at detecting both lane lines. In particular it captures very well the right hand lane line which is challenging due to the breaks between lines. The luminosity thresholding is shown in blue. At first glance, this appears to be less useful than red thresholding as less pixels are activated. However, the L channel thresholding serves an important role in detecting parts of the yellow lane line (left lane line) that were missed by the red thresholding. This can be seen in those pixels that are pure blue (detected only by L channel) as opposed to purple (detected by both channels). Without the luminosity thresholding, there were occasional problems in detecting the left hand lane line and sometimes the detected lane would jump to the central barrier. Therefore, the L channel serves an important role in supporting the red channel thresholding.

Ultimately, we produce a "binary thresholded image" where pixels are either 0 or 1. To do this we stack our red thresholded image with our luminosity thresholded image and a blank image to get a three channel image. We then use OpenCV to map this to a grayscale image such that any pixel that was activated by *either* the red channel or the luminosity channel remains active. This is shown in the final image on the right below. We see it does a good job of detecting both lane lines and not much else. It is important that we don't detect too much else since if we have a very noisy binary image, the detected lane line will jump around all over the place.

![alt text][image2]

#### 3. Region masking
As mentioned above, we want to minimise noise to avoid the detected lane line jumping too much. The particular thresholds that we have chosen already do a pretty good job of this but we can improve things further by masking a particular region of the image to block out activated pixels that we know are unlikely to be anything to do with the lanes e.g. in the rightmost image above, there is a white patch to the left of the lane line that could potentially cause confusion (it doesn't actually). Note that this is a little bit of a cheat since it would not be practical in real life - it is only suitable to this video since the car never changes lane. If the car is changing lane then we need to be able to track a wider range of pixels.

We can see the benefits of region masking below:

![alt text][image3]


#### 4. Perspective transform

Perspective is the phenomenon whereby objects in an image appear smaller the father away they are from the viewpoint. This is seen by parallel lines appearing to converge towards a single point; when viewing a scene from different perspectives, this point of convergence will change.

A perspective transform is a mathematical operation (affine transformation by matrix multiplication) that transforms an image to view it from a different perspective.

One of the tasks in this project is to establish the curvature of the lane line. This will require us to make a perspective transform. The reason for this is that in the original image, the left hand lane line appears to curve to the left whilst the right hand lane line appears to curve to the left when in fact we know that they should both be parallel to one another. By transforming the image to a bird's eye view, we are able to see both lines as parallel. From this perspective, both lane lines should agree on the curvature of the road.

To actually make this perspective transform, we know that in our bird's eye view, the lane lines should appear as a rectangular region. Choosing a simple example image from the original camera perspective with an uniterrupted view of a straight section of road, we can manually identify four source (`src`) points representing the vertices of a trapezoidal region enclosed by the lane lines. We then choose four destination (`dst`) points that will form the vertices of a rectangular region in our transformed image. We can then use the OpenCV function `cv2.getPerspectiveTransform` to find the matrix that performs the necessary affine transformation. This will transform the lane lines from the boundary of a trapezoidal region to a rectangular region which thus guarantees they are parallel. We can store this matrix in a pickle file and use it for all subsequent images.

I used the following source and destination points:

| Source        | Destination   | 
| :-----------: |:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1106, 720     | 960, 720      |
| 700, 460      | 960, 0        |

You can see the perspective transform in action below:
![alt text][image4a]
Once we have done the perspective transform once, we can take any binary image and perform region masking so that the remaining activated pixels are only in the region surrounding the lane lines. We then load the above affine matrix from the pickle file and apply the perspective transform to bird's eye view. Doing so, we arrive at an image that looks like the following:
![alt text][image4b]
Note that the red regions are drawn only for this writeup and are not normally included. 

#### 4. Identifying lane pixels and fitting a polynomial equation

The next step is to identify the pixels belonging to the left and right lane lines and fit a polynomial equation to each of them separately.  Our first task is to identify the approximate location of the lane lines. We can do this by summing the pixel values (0 or 1) along each column to produce the following histogram:
![alt text][image5a]
We will attempt to fit a polynomial of degree at most 2 to each of the lane lines. In the case when the lines are very straight, the leading coefficient will of course be extremely small. As it happens, we shall in fact fit this quadratic equation to x(y) rather than y(x) since such horizontally orientated parabolas better capture the trajectories of lane lines from this perspective. The histogram above can be used to infer the x-intercept of each of these parabolas.

We will then subdivide the vertical range of the image into 9 "sliding windows" (shown with green bounding boxes below). The windows each have a width of 200 pixels and the two bottom windows will start centred on the x coordinates of the two histogram peaks. We then count the number of activated pixels within each these two bottom windows and if they exceed a certain threshold, we shall position the windows on the next layer to be centred on the mean x coordinate of the activated pixels from the previous layer. By continuing this procedure, our 9 windows are able to slide horizontally to track the locations of the activated pixels and thus track the locations of the lane lines. All activated pixels contained in the left hand sliding windows are coloured red and all activated pixels contained in the right hand sliding windows are coloured blue. Activated pixels outside the sliding windows remain white.

We can then apply the NumPy function `np.polyfit` to the lists of left, or red, y and x coordinates to obtain a best-fit parabola for the left lane line. This can be repeated for the lists of right, or blue, y and x coordinates to get the best-fit parabola for the right lane line. In actuality, `np.polyfit` returns a list of coefficients $[A,B,C]$ but we can use these to obtain the corresponding quadratic equation,

$x = Ay^2 + By + C$.

For display purposes, we draw the two parabolas in yellow on the image below (note we won't usually do this):

![alt text][image5b]

#### 5. Calculate lane curvature and vehicle offset

Our final step is to compute the curvature of the lane and the position relative to the centre of the road (offset) with respect to the current location of the car.

Let's start with the curvature, or rather, the radius of curvature at a given point, $p$, on the curve:

![alt text][image6a]

The idea is that we position a circle so that it osculates ("kisses") the circle at $p$.In these circumstances, the curve and the circle willnecessarily have the same tangent and curvature at $p$. The radius of curvature is simply the radius of the *osculating circle*. This gives one means of measuring curvature. 

How do we actually measure this though?

The curvature of a plane curve at any point is the limiting ratio of $d \theta$, an some infinitesimal segment of the curve



If we take an infinitesimal segment along the curve with length $ds$. Let $d \theta$ be the infinitesimal angle (in radians) between the tangents at opposite ends of this segment. We know from the definition of arclength that $ds = r d \theta$ where $r$ is the radius (of curvature). Thus we get that $r = \frac{ds}{d \theta}$.

Now we know from Pythagoras that 

$ds=\sqrt{dx^2+dy^2}$.

We also know that $M_\text{tangent}=\frac{dy}{dx}=\tan{\theta}$,  where $\theta$ is the angle the tangent line makes with the x-axis.

It follows from some straightforward calculus that

$r = \frac{ds}{d \theta} = \frac{dx}{dx} \cdot \frac{dx}{d \theta} = \frac{\frac{ds}{dx}}{\frac{d \theta}{dx}} = \frac{\sqrt{1 + \left( \frac{dy}{dx} \right)^2}}{\frac{d \left( \tan^{-1} \frac{dy}{dx} \right)}{dx}} = \frac{\left[1+ \left(\frac{dy}{dx}\right)^2\right]^{3/2}}{\frac{d^2y}{dx^2}}$.

Bear in mind that in our case, we are dealing with a curve $x(y)$ rather than $y(x)$ and so we should swap their roles accordingly to obtain the definition

$R_\text{curvature} = \frac{\left[1+ \left(\frac{dx}{dy}\right)^2\right]^{3/2}}{\frac{d^2x}{dy^2}}$.

Simple differentiation reveals that if $x=Ay^2+By+C$, then 

$R_\text{curvature}(y) = \frac{(1+(2Ay+B)^2)^{3/2}}{|2A|}$. 

Note the absolute value on the denominator prevents us obtaining a negative radius. We need to pick a $y$-value at which to evaluate this qunatity and it makes sense to evaluate it at the maximum y-value (bottom of the image) since this corresponds to the car's current location which is, of course, what we are most interested in. This quantity is very easily computed using the coefficients returned in our list of parabola coefficients from `np.polyfit`.

It is worth commenting that the tighter the corner, the smaller the radius of curvature. Consequently, a straight stretch of road will have a very large radius of curvature; indeed, for a perfect straight line, this should be infinite.

The second thing we wanted to do was compute the vehicle position relative to the centre of the road. Again, this is fairly simple. The centre of the road is taken to be the mean of the two histogram peaks i.e. the mean of the two x-intercepts (we use the x-intercepts since this corresponds to the car's current location). Assuming the camera is mounted exactly in the middle of the dashboard, we can then compute

$\text{offset} = \frac{\text{width}}{2} - \frac{C_\text{left} + C_\text{right}}{2}$ .

Finally, we need to scale these values so that they return a curvature and offset in terms of metres rather than pixels. We will assume  a scale of
 
`ym_per_pix = 30/720` (30 metres for every 720 vertical pixels)

`xm_per_pix = 3.7/700` (3.7 metres for every 700 horizontal pixels)

It is then just a matter of correctly scaling the offset and curvature from pixles to metres.


#### 6. Projecting back onto the road: inverse bird's eye projection

When we obtain the matrix that transforms camera images to bird's eye views, we can also obtain the inverse matrix that maps from a bird's eye view back to the original camera perspective. We will again pickle this matrix as we will be using it a lot.

The idea is that once we have identified the parabolic equations of our two polynomials, we can plot them on our bird's eye image and use `cv2.fillPoly` to colour the region between them in green. We will then apply the inverse bird's eye transformation to this and recover the original image overlaid with a green area corresponding to the lane region identified.

Applying this whole pipeline to an input image, we obtain the following output:

![alt text][image6b]

Notice that I have provided the original image, the bird's eye perspective, the lane detection and the measurements of curvature and offset at the top.

---

### Pipeline (video)

By splitting an input video into individual frames, we can apply the above pipeline frame-by-frame and then combine the output images into a new video. 

Although a GIF of the final output is shown at the top of this file, here's a [link to my full video result](./project_output_video.mp4)

---

### Discussion

My code works very well on the sample video. However, it would likely need improvement for dealing with the following situations:

* vehicle changing lane (probably need to remove region masking)
* extreme corners e.g. at a T-junction (parabola would struggle to capture this)
* sudden changes in brightness/shadows - the thresholding part of my code does quite well at this on this particular sample video but experimenting with other videos leads to some problems. This needs to be made more robust.


