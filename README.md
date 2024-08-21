# Handwritten Digit Recognition and Neural Networks

### Description
The programs written in this repo are different ways to recognize images of handwritten digits and classify them. The majority of code was a personal
implementation of a barebones convolutional neural network. I spent numerous hours working out the math behind backpropogation and writing code to facilitate
the algorithm. The basic implementation was 90-95% accurate on the test data. The remaining time was spent improving the performance of my code, both in terms of 
memory and time optimization, as well as using validation data to tweak parameters in order to improve accuracy. My best run achieved >99% accuracy; however, I'd like
to emphasize that **this project was not meant to build the best handwritten digit classifier out there**. This project's major goal was to teach me core principles
behind neural networks and identify different design levers ML engineers play with when they consider solving problems using neural networks. Lastly, I tried to 
apply some of these principles to write different neural networks using a high-performance ML library *TensorFlow*. This is not an endorsement of this library
over any other libary - I picked one that was recommended to me by a friend.

### Credits
This project avoids one of the major machine learning problems - namely all of the data was created, processed, labelled, and provided for me by the 
*mnist* database. 

This project was inspired by and mostly designed by following *Michael A. Nielson, "Neural Nets and Deep Learning", Determination Press 2015*.
While I took some of my own twists on the implementation details, the structure of the code was taken almost directly from the book. The code here 
may not look exactly like the code featured in the book because Nielson leaves certain challenges for students to update their code to improve 
performance. I would like to express my sincerest gratitude for such an excellent publication put online for free for students to access.
Machine learning was always a course I was interested in taking during my undergraduate, but never had the time to pursue it. This book was a lovely
primer on the topic and made discussions about more advanced concepts much more accesible.
