# adversarial-attacks-demonstration

###### Quick Start -> open the demo.ipynb
<font size="3">
This Notebook aims to demonstrate one of the inherent weaknesses of using convolutional neural networks (CNN's) in safety critical systems such as security systems, autonomous driving applications and more. The notebook offers a walk through of the steps and code necessary for you to understand the concepts involved and to try out the idea for yourself. Please feel free to re-use any code you see in this document.

* <pre> demo.ipynb   ->  demonstration notebook to follow along</pre> 
* <pre> image.py     ->  supporting python file that provides the Image class and additional functionality I created</pre> 
</br>
</font>



## The weakness
<font size="3">**The Architecture:**
CNN's and NN's rely on precise weights that have been calibrated through the propagation of training errors to correct these prediction weights that are stored in the neurons of the hidden layers in the networks.</font>
</br></br>
<font size="3">**The Hypotheses:**
If we had access the the model (and therefore the neuron weights) used in these safety critical systems we could potentially exploit their over reliance on particular neurons and their associated weights to adjust the model predictions to our benefit.</font>
</br></br>
<font size="3">**The Approach:**
This notebook uses the famous ResNet-18 dataset and model as an example of how these weights can be used to our advantage as well as some new arbitrary and unseen images for the model to make predictions on.</font>


