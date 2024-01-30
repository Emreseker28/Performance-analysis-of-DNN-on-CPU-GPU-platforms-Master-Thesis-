<h1>Aim</h1>
This thesis’ aim is to find how much of a difference in performance when using a Graphics
Processing Unit (GPU) and a Central Processing Unit (CPU) when training a Deep Neural
Network (DNN) model. To achieve the results, machine learning model will run on GPU and
CPU under same conditions, multiple times. After that, the machine learning model will be
altered slightly to see if there is any impact on the performance difference. 

<h1>Platform Characteristics</h1>
This part of the paper is about the hardware that has been used in this thesis. Both physical
as well as cloud service (Google Colab) hardware have been used. It is important to mention
that all devices that have been used are assumed to perform at their highest utilization without
any other program or service interfering.<br />
<h3>Hardware Specifications</h3>

Name | CPU | GPU | Google Colab 
--- | --- | --- | --- |
Processor | Intel® Core(TM) i5-8350U | Nvidia GeForce GTX 980 | Nvidia Tesla T4
Number of Cores | 4 | 2048 | 2560
Frequency | 1.70 GHz (3.60 GHz with Turbo Boost) | 1127 MHz (Base), 1216 MHz (Boost) | 585 MHz (Base), 1590 MHz (Boost)
Memory | 24 GB DDR4 2400 MHz | 4 GB GDDR5 1753 MHz | 16 GB GDDR6 1250 MHz

<h3>Libraries used:</h3>

Name | Version
--- | --- |
Python | 3.10.9
Pandas | 1.5.3
NumPy | 1.23.5
Keras | 2.10.0
TensorFlow | 2.10.1
Scikit-learn | 1.2.1
MatPlotLib | 3.7.0
CuDNN | 8.1.0.77
CudaToolkit | 11.2.2
SciPy | 1.10.0
Conda | 23.3.1

<h1>Model Design</h1>
A basic-level representation of the machine learning model that is being used. <br />

![image](https://github.com/Emreseker28/Master-Thesis/assets/54375145/c3c01341-ee4e-452d-9ced-e93e973d0353)

<h1>Results</h1>
To get a better idea about the performance difference between CPU and GPU, several tests have been conducted. To eliminate the possibility that the background tasks from the 
operating system may interfere with the results, all test conditions have been repeated 5 times.
And the median values of those test results will be the main factor that will be focused on
(mean value can also be used, but mean, by its calculation method, is more viable to the
outliers compared to the median value).

<h2>Batch Size</h2>
One of the tests consisted of changing the batch size on the training options. Batch size
represents the number of training examples utilized in one iteration. Generally, this parameter
favors the hardware that has the highest core count. And as can be observed from tables below, the GPU is faster than the CPU, as expected. There is up to 4.71x performance
increase between these devices. In the worst scenario, GPU is still 1.11x percent faster than
the CPU. The graphical representation of these performance differences can be observed in
figure below. These improvements can be significant depending on the view angle. For the
dataset this small, these changes may not mean much. But if it is compared to a much bigger
dataset, 4.71x performance difference could mean saving up to hours of training time. This
performance difference can also be associated with the advantages that come with
parallelism. As mentioned before, GPU has more core count than CPU. Therefore, increasing
the batch size could allow the program to utilize the cores in a more efficient way.	

Results of the batch size changes and comparison of CPU vs GTX 980	

Batch Size | 4 | 8 | 16 | 32 | 64
--- | --- | --- | --- | --- | --- |
CPU | 96.857 | 61.883 | 32.950 | 20.570 | 24.122
GTX 980 | 22.200 | 13.134 | 9.441 | 7.559 | 6.660
Speed Up | 4.36x | 4.71x | 3.5x | 2.72x | 3.62x

Results of the batch size changes and comparison of CPU vs Tesla T4

Batch Size | 4 | 8 | 16 | 32 | 64
--- | --- | --- | --- | --- | --- |
CPU | 96.857 | 61.883 | 32.950 | 20.570 | 24.122
T4 | 44.657 | 29.775 | 21.637 | 18.438 | 16.721
Speed Up | 2.16x | 2.07x | 1.52x | 1.11x | 1.44x

![image](https://github.com/Emreseker28/Master-Thesis/assets/54375145/a0e8198d-a5d9-481b-bb9f-9753af317516)

<h2>New Dense Layers</h2>
After observing the performance differences in different batch sizes, it is decided to add new
layers to the machine learning model to see if there is any impact on the performance. First,
the dense layer number has been increased. Tables below consists of the result of the
tests done with different dense layer numbers. It is clear that GPU has an advantage over the
CPU. The difference between the CPU and the fastest GPU is between 3x and 4.15x, which
aligns with the previous tests results. Figure below shows the performance difference in a chart
graph. It is important to mention that adding 1 new dense layer corresponds to a total of 3
dense layers in the machine learning model.

Results of the new dense layers and comparison of CPU vs GTX 980	

New Dense Layers | 1 | 2 | 3 | 4 | 5
--- | --- | --- | --- | --- | --- |
CPU | 33.591 | 30.509 | 30.439 | 29.791 | 34.688
GTX 980 | 8.098 | 8.720 | 9.257 | 9.836 | 10.404
Speed Up | 4.15x | 3.5x | 3.3x | 3x | 3.33x

Results of the new dense layers and comparison of CPU vs Tesla T4

New Dense Layers | 1 | 2 | 3 | 4 | 5
--- | --- | --- | --- | --- | --- |
CPU | 33.591 | 30.509 | 30.439 | 29.791 | 34.688
T4 | 18.436 | 21.245 | 21.794 | 22.085 | 22.983
Speed Up | 1.82x | 1.43x | 1.4x | 1.35x | 1.5x

![image](https://github.com/Emreseker28/Master-Thesis/assets/54375145/20901e0a-7959-4f3a-ac6a-e1a618376d04)

<h2>New Convolutional Layers</h2>
Lastly, the performance difference after adding more convolutional layers has been tested.
This test has a similar effect to the adding more dense layers test. Same as before, up to 5
new convolutional layers have been added to the machine learning model and tested. Tables below contains the test results; a graphical representation can be found in figure below.
An important thing to mention is
that in these tests, CPU and as well as the T4 GPU gave inconsistent results. Although Google
claims that T4 GPU on the Google Colab servers are dedicated to the user in that current
session, these results states otherwise. It is thought that there was some interference (it may
be the GPU that is being used has been shared with other users) or the GPU on that session
when the tests were being conducted had some performance issues. 

Results of the new convolutional layers and comparison of CPU vs GTX 980	

New Convolutional Layers | 1 | 2 | 3 | 4 | 5
--- | --- | --- | --- | --- | --- |
CPU | 28.770 | 24.728 | 25.009 | 30.950 | 28.091
GTX 980 | 8.140 | 8.682 | 9.184 | 9.737 | 10.247
Speed Up | 3.54x | 2.85x | 2.72x | 3.18x | 2.75x

Results of the new convolutional layers and comparison of CPU vs Tesla T4

New Convolutional Layers | 1 | 2 | 3 | 4 | 5
--- | --- | --- | --- | --- | --- |
CPU | 28.770 | 24.728 | 25.009 | 30.950 | 28.091
T4 | 19.237 | 18.759 | 20.053 | 19.211 | 19.645
Speed Up | 1.5x | 1.32x | 1.25x | 1.61x | 1.43x

![image](https://github.com/Emreseker28/Master-Thesis/assets/54375145/d8b40d12-bfd6-4442-9136-00fda41430d7)

<h2>Accuracy</h2>
Even though it has been stated that the accuracy of the machine learning model is not
important (or getting the highest accuracy is not in the scope) for this thesis, it is good practice
to mention it. In table below, the accuracy results of one of the tests that have been conducted
can be found. Even though the accuracy metrics’ results are not the highest, they are in the
range of what can be called “acceptable”. Of course, aiming for accuracy higher than 90% is
the aim for the most machine learning practices, this paper does not aim for the highest
accuracy

Iteration | MSE | MAE | Pearson | Spearman | R^2
--- | --- | --- | --- | --- | --- |
1 | 2.207 | 0.659 | 0.770 | 0.566 | 0.593
2 | 2.175 | 0.659 | 0.733 | 0.539 | 0.537
3 | 2.216 | 0.675 | 0.716 | 0.485 | 0.513
4 | 1.915 | 0.634 | 0.799 | 0.538 | 0.639
5 | 2.116 | 0.659 | 0.776 | 0.553 | 0.602
Mean | 2.1258 | 0.6572 | 0.7588 | 0.5362 | 0.5768
Median | 2.175 | 0.659 | 0.770 | 0.539 | 0.593

<h1>Conclusion</h1>
In conclusion, the comprehensive testing conducted to assess the performance difference
between CPU and GPU in machine learning tasks has yielded valuable insights. The focus on
median values and repeated testing to mitigate background interference contribute to the
robustness of the findings. <br> <br>
The evaluation of different batch sizes revealed a notable advantage for GPU over CPU, with
performance improvements ranging from 1.11x to 4.71x. This increase in speed, especially
with larger datasets, underscores the efficiency of GPU parallelism. However, the resourcebounded nature of batch size, particularly limited by GPU memory, highlights a trade-off
between computational speed and available resources. The examination of additional dense
layers demonstrated that GPU consistently outperforms the CPU, with a performance
difference ranging from 1.35x to 4.15x. Notably, the impact of layer addition on performance
was less than anticipated, suggesting that dataset complexity and neuron count play crucial
roles. Similarly, the introduction of more convolutional layers exhibited GPU superiority,
offering up to a 3.54x performance increase over the CPU. Despite some inconsistency in T4
GPU results, the overall trend reinforces the clear advantage of using GPU for machine
learning model training. While accuracy was not the primary focus of the study, an example
table presented acceptable results for the chosen metrics. The median values indicated
consistent model performance across iterations, providing reasonable accuracy in capturing
relationships within the data. <br> <br>
In summary, the findings underscore the significance of GPU acceleration in machine learning
tasks, especially in scenarios where computational efficiency is important. The study's
meticulous methodology and detailed results contribute valuable insights for practitioners
seeking to optimize their machine learning workflows.
