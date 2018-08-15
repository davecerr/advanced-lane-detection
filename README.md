# advanced-lane-detection

![Alt Text](https://media.giphy.com/media/4EFsn95JAPySiR03Yo/giphy.gif)

In this repository we create a machine learning pipeline capable of performing lane detection for a self-driving car using a videostream of dashcam footage. To each frame of the video we remove camera distortion effects, identify candidate lane line pixels using colour channel thresholding, perform a perspective transform to a bird's eye view so that the lane lines appear parallel, use a "sliding window" approach to identify the pixels belonging to each lane, fit a quadratic equation to both the left and right lane lines, and finally measure the curvature of the lane and the vehicle's offset from the centre of the lane using the coefficients of the quadratic. These measurements can then be used to inform steering decisions etc.

I have written a post explaining the problem and the techniques involved in the solution at [https://daviderrington.net/2018/07/23/the-mathematics-of-perspective/](https://daviderrington.net/2018/07/23/the-mathematics-of-perspective/).

