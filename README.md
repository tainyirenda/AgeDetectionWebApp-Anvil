<h1 align="center">Hi 👋, I'm Tai</h1>
<h3 align="center"> Welcome to my Age Detector Project.</h3>
<br/>
🌱 Check out how I used a Deep Learning solution with a Convolutional Neural Network (CNN) for age detection. Deployed to Anvil Web App so you can upload your photo and see if it is detected as young, middle-aged, or old. (Disclaimer: I hold no responsibility for age classification disappointment 😂) <br/>

# Age Classification Model Hosted on Anvil Web App

### **Introduction**

This report delves into the development of a Convolutional Neural Network (CNN) for Age Detection, categorizing faces into 'Middle', 'Young', and 'Old'. The extensive dataset, consisting of 19,906 facial images, posed a challenge and made CNN the optimal choice due to its proficiency with large datasets. CNN models have been widely used in computer vision since 2012 as part of Deep Neural Network Architecture, outperforming other techniques in classification (Krizhevsky et al., 2012).
The relevance of age detection extends beyond industrial applications; it is crucial in security and public safety measures. For example, age detection can be employed in surveillance systems to enhance security protocols at public spaces or identify potential age-specific safety concerns. Moreover, in healthcare, understanding age demographics from medical images could contribute to personalized treatment plans. However, age detection presents its own set of challenges. Facial features can have significant variations due to factors such as lighting conditions, image quality, and diverse ethnicity variances, gender and facial expressions (Katherick, 2018) . Additionally, the characteristic partiality in placing individuals into broad age groups introduces challenges. These challenges enforce the need for development of robust models capable of discerning subtle age-related features among variations, making age detection a required, yet complex, task in the field of computer vision.
Nonetheless, age detection is a valuable part of computer vision that plays a fundamental role in a variety of applications with greatly widespread implications. The ability to accurately classify facial images into age categories holds significant value in various fields. This report explores not only the CNN model but also incorporates data augmentation and bounding box techniques to enhance object recognition, model performance, and counter overfitting. Having a large dataset for training allows the model to better learn and recognize objects, considering the variability objects can present in real-world settings.
<br/>
<br/>
<br/>
**Packages**

Packages used throughout were as follows: <br/>
•	NumPy (for dataset arrays) <br/>
•	Pandas (for working with data frames) <br/>
•	Matplotlib (for plotting) <br/>
•	TensorFlow (with tfds dataset management) <br/>
•	PIL (for image augmentation) <br/>
•	Keras (for image preprocessing and model layers) <br/>
•	SciKit Learn (train/test split of dataset for model selection and evaluation metrics) <br/>
•	Pydot (for model visualisation) <br/>
•	Glob (for file pattern matching and file directory handling) <br/>
•	Os (for file and directory handling) <br/>
•	Cv2 (for functionality of image processing)<br/>
•	Anvil (for server connection and model deployment)
<br/>
<br/>

### **The Model** <br/>
In this code, we define a simple CNN model with several convolutional layers followed by max-pooling layers, and then fully connected layers for classification. The final layer uses SoftMax activation and a loss function of sparse categorical cross entropy.  CNN is designed for image classification into three age categories using three Convolutional layers followed by fully connected layers. The model uses ReLU activations for feature extraction and SoftMax activation for the final classification. <br/>

The first layer is a Convolutional layer with 32 filters, each of size (3, 3), using the ReLU activation function. It takes input images with a shape of (150, 150, 3), where 3 is the number of colour channels (RGB). This layer is followed by a MaxPooling layer with a pool size of (2, 2), which reduces the spatial dimensions. <br/>

The next 2 layers are another convolutional layer with 64 filters and a (3, 3) kernel, followed by a MaxPooling layer. <br/>

Following is another convolutional layer with 128 filters and a (3, 3) kernel, followed by MaxPooling. <br/>

The flattening layer flattens the output from the previous layers into a 1D array, preparing it for the Dense layers. <br/>

Subsequently, two Dense layers follow the flattened layer. The first Dense layer has 512 neurons with ReLU activation, and the final Dense layer has 3 neurons (for the three age categories) with a SoftMax activation function. SoftMax is used for multi-class classification, converting the output into probability scores for each class. <br/>

The model compilation consists of the Adam optimizer, sparse categorical cross entropy as the loss function (suitable for multi-class classification tasks where the labels are integers), and accuracy as the evaluation metric. The layers and parameters are as follows: <br/>

•	Input Layer: <br/>
Type: Conv2D <br/>
Filters: 32 <br/>
Kernel Size: (3, 3) <br/>
Activation Function: ReLU <br/>
Input Shape: (150, 150, 3) - Assumes input images are 3 channel images with size 150x150 pixels.
<br/>

•	Max Pooling Layer: <br/>
Type: MaxPooling2D <br/>
Pool Size: (2, 2) <br/>
This layer reduces the spatial dimensions of the representation.
<br/>

•	Convolutional Layer: <br/>
Type: Conv2D <br/>
Filters: 64 <br/>
Kernel Size: (3, 3) <br/>
Activation Function: ReLU <br/>

•	Max Pooling Layer: <br/>
Type: MaxPooling2D <br/>
Pool Size: (2, 2)
<br/>

•	Flatten Layer: <br/>
This layer flattens the 2D output from previous layers into a 1D array before feeding it into the fully connected layers.
<br/>

•	Dense (Fully Connected Layer):<br/>
Type: Dense layer <br/>
Units: 512 (neurons) <br/>
Activation Function: ReLU <br/>

•	Output Layer:<br/>
Type: Dense <br/>
Units: 3 (neurons) <br/>
Activation Function: Softmax – for multi-class classification task with three classes
<br/>

•	Compilation:<br/>
Optimizer: Adam<br/>
Loss Function: Sparse Categorical Cross entropy – for integer-encoded class labels<br/>
Metrics: Accuracy
<br/>
<br/>
<br/>

**The Dataset/Data Pipeline** <br/>
The dataset is comprised of a file sub-directory called ‘Train’ which contained 19,906 images of faces. These images were stored with variability of augmentations. Images had a wide range of variability i.e. gray-scaled facial image, colour facial images, rotated images, facial images taken from different angles and different quality facial images e.g. some blurry. The variability of the facial images dataset gave a better representations of how photos may be in real-world settings, allowing for better generalisation.  
The folder directory contained a CSV file named ‘train.csv’ which contained a table with two columns, one with a list of the images by name, and one with a list of category labels of ‘young’, ‘middle’ and ‘old. There were 10804 images belonging to the ‘Middle’ class (54.3%), 6706 images belonging to the ‘young’ class (33.7%), and 2396 images belonging to the ‘Old’ class (12%). 
<br/>
<br/>
<br/>
### **Methods** <br/>

**Dataset:** <br/>
Both the CSV classes and the facial images were combined linked by the ID columns containing the image file names to respectively link the images to their labels. The images were then resized to 150x150, set with three channels. After being successfully combined and composed into a TensorFlow dataset, the category string labels were mapped and converted into integers labels, similar to one-hot encoding, in preparation for the model activation function for multi-class classification. 
The TensorFlow dataset was split into an 80/20 division. For the training set of the dataset 80% was used for the model, and for the testing set of the data 20% was used for the model. Before passing the split data through the model, autotune object was created for both the training dataset and testing dataset. This was performed to allow loading and processing of each element of the dataset and for batching images to be processed simultaneously. This improves training accuracy and with prefetch batching, loading and training time are both improved for the model performance, additionally aiding in efficient processing during model evaluation.<br/>
<br/>
**Augmentation Techniques:** <br/>
Considering that the nature of the dataset contains wide variability and some augmentations already, the dataset was only mildly augmented using random brightness, random contrast, random saturation and random hue. This was only implemented on the training dataset, whilst the testing dataset was left in its raw state for comparison. Augmentation techniques were implemented using TensorFlow packages, NumPy and PIL. The combination of these libraries allows for efficient data augmentation, a crucial step in enhancing the performance and generalization of machine learning models, particularly in computer vision tasks.<br/>
<br/>
**Bounding Boxes:** <br/>
Bounding boxes provide a simple and intuitive way to represent and describe the location of objects within an image. To implement bounding boxing, TensorFlow operations were used to read the image file, decode it as a JPEG image, convert its data type to float32, and resize it and return the pre-processed image. Then, using the OpenCV library cv2, the images were converted into gray-scale with the faces in the image detected using a pre-trained Haarcascades face detector. Finally, bounding boxes were drawn around detected faces in the original image with a returned list of bounding box coordinates. The functions, together, allow for reading, preprocessing, and detecting faces in images to fit to the bounding box.<br/>
<br/>
**Model Deployment:** <br/>
After several model runs, the model for implementation was saved using the ‘model.save()’ method to a specified file path. After this, the model was deployed using Anvil. Anvil is a platform that allows building of web applications using only Python code. Deploying a machine learning model to Anvil typically involves creating a web app that interacts with a specified model.
Anvil contains a drag-and-drop interface builder to design the user interface of a web app. It allows for addition of input forms, buttons, and any other elements needed to interact with your machine learning model. In the specific design for this report; an uploader component which allows image files, an image component to display the image and a label component to display the predicted result of the passed imaged were used for design.<br/>

Once the Graphical User Interface (GUI) of the app had been designed, deployment continued using Anvil's deployment options. To do this, use of Python code in the backend of the design was used. Anvil provides hosting for your web apps, making it accessible over the internet using a coding script in the back end. The code gave functionality to the components on the design page using a callable function that was designed in the notebook used as source for the workings of the model.<br/>

The callable function in the notebook produced for this report, allowed for connectivity to the server, to be able to implement the created and tested CNN Model that was then saved for use. The connectivity to the serve for the GUI to function linked using and uplink connection to anvil using the Anvil.Server module, and then passed the callable function definition which defined what model to use with images passed and how to handle images passed.<br/>

Once deployed (using publishing in Anvil), a public set URL for the published Anvil app was generated so that users can access the web application through this URL. This enabled users to now interact with the CNN model through the web app. They can input data, trigger predictions, and visualize the results in the GUI.
<br/>
<br/>
<br/>

### **Results/Evaluation** <br/>
For the CNN model run, the model achieved a final accuracy of around 93% after 10 epochs with loss of 0.1975 on the training dataset. On  the test dataset, the model achieved a final accuracy of 92% with loss of 0.2105. <br/>

The Train dataset accuracy reflected the performance of the model on the 80% of the data used for training after the final epoch. The values increase over epochs, indicating that the model is learning the patterns from the training dataset well.The increasing values on the training set at each epoch during training indicated that the model is learning from the majority of the data. <br/>

The Test dataset accuracy reflects the accuracy on the separate 20% of the data not seen during training in figure 10. It shows how well the model generalizes to new, unseen data. The values generally increase, showing that the model is also generalizing well to unseen data as shown in figure 9. Furthermore, from the classification report it is shown that the model has high precision, recall, and F1-score for all three classes, indicating good performance. Overall accuracy is around 92.67%, which is the proportion of correctly classified instances among the total instances in the test set. The confusion matrix provides a detailed breakdown of correct and incorrect predictions for each class. From this, it can be concluded that the model seems to perform well on the test set, achieving good accuracy and balanced performance across different classes. This was performed only on the test set to validate further that the model generalises well on new and unseen data, rather than data it has already seen before and give and idea on the ability of the model to handle new instances. <br/>

In summary, the training and testing accuracies increase over epochs, which is an indication of good performance of the model. The overall accuracies are reasonably high, suggesting that the model is performing well on both the training and testing datasets. This indicates good generalization of the model when dealing with new data it has not seen before and good learning from the majority of data. <br/>
<br/>
 
**Results Comparing Model Without Pre-Processing, with Augmentation and Bounding Boxes** <br/>

*No Pre-processing Model:* <br/>
The accuracy exhibits a promising upward trend, starting at 61.9% in the first epoch and steadily increasing to approximately 95.7% by the tenth epoch. The model demonstrates effective learning without additional pre-processing steps. The loss steadily decreases from 0.8491 in the first epoch to 0.1275 by the tenth epoch. This consistent reduction indicates effective learning and improvement over successive epochs.
<br/>

*With Augmentation:* <br/>
Augmentation introduces variability in the training data, resulting in a lower initial accuracy of around 57.6%. However, the model improves over epochs, achieving approximately 82.0% accuracy by the tenth epoch. Augmentation's impact on diversity contributes to the observed accuracy fluctuations. Despite higher initial loss values compared to the no pre-processing scenario, the model still demonstrates improvement. The final loss is 0.4226, reflecting a positive impact from the introduced variability in the training data.
<br/>
<br/>
*With Bounding Boxes:* <br/>
Utilizing bounding box information yields a comparable initial accuracy of around 62.5% compared to the no pre-processing case. The model continues to learn effectively, reaching approximately 93.4% accuracy by the tenth epoch. This suggests that incorporating bounding box information positively influences the model's learning process. The model incorporating bounding box information follows a decreasing loss trend, starting at 0.8327 and reaching 0.1887 by the tenth epoch. This pattern indicates that incorporating bounding box information enhances the model's ability to achieve a lower loss over the training period. 
<br/>
<br/>
<br/>
 
### **Conclusion** <br/>
The model with no pre-processing achieves the highest accuracy, but it might benefit from additional techniques or adjustments to reach even higher performance. The augmented model, while starting with lower accuracy, shows improvement and might generalize better to unseen data due to data augmentation. However, there was not a significant improvement seen in the augmented model in comparison to the model without augmentation. On the other hand, the model with bounding boxes performs well, leveraging additional spatial information. It achieves high accuracy and a relatively low loss, indicating the effectiveness of incorporating bounding box data. This model suggested improved slightly in accuracy.

While data augmentation is generally beneficial for training robust models, there are scenarios where certain augmentation techniques might not improve or could potentially decrease model accuracy. On the other hand, augmentations simulate real-world conditions, helping the model be more adaptable to challenges it may face. Providing multiple techniques allows the model to be trained on a wider variety of data, resulting in a more diversified dataset for better generalization to unseen data. As suggested in the results of the model run with no augmentation or pre-processing techniques, the model indicated generalising well to unseen data before any experimentation with other augmentation techniques or implementation of bounding boxes. Bounding boxes can be applied to a wide range of object types, including people, animals, vehicles, and more. They are versatile and can be used in both single-object and multi-object scenarios, which would be useful in imagery examples that include more than one person in real-world settings. However, Irregularly shaped objects may have a significant portion of the bounding box containing non-object regions and in cases where objects are close or overlapping, bounding boxes may overlap, leading to challenges in determining the precise boundaries of individual objects.
In summary, each pre-processing approach influences the model's performance differently. While augmentation introduces diversity at the expense of initial accuracy, bounding box information proves beneficial in enhancing both accuracy and the model's ability to minimize loss over successive epochs. The choice of pre-processing strategy should be carefully considered based on the specific requirements and characteristics of the given task. 
<br/>



### **Recommendations** <br/>
The model with implemented bounding boxes is recommended for its higher accuracy and better generalization, especially if the dataset is limited. However, this can also depend on the dataset and the model architecture. For future recommendations, it’s always good practice to validate conclusions through multiple experiments and datasets. As Bounding boxes do not provide detailed information about the interior structure of objects and focus on specifying a bounding region, there can be loss of fine-grained information. For more advanced applications, techniques like instance segmentation or keypoints estimation may be considered.
If good model performance is not achieved with other data, methods such as using validation set performance, testing on unseen data, and possibly fine-tuning the model for better generalisation can be considered. Possible techniques can also entail introducing drop put layer to the model architecture. Dropout is a powerful regularization technique that helps prevent overfitting by randomly dropping out neurons (introducing noise) during training, encouraging the network to learn more robust and generalizable features. Neurons become less dependent on specific input features, promoting this learning. Additionally, adding a penalty term to the loss function based on the magnitudes of the weights could also optimise model performance. L1 regularization encourages sparsity, while L2 regularization penalizes large weights.
Deeper neural network depth is also something to consider as it can significantly improve model performance (Kim et al, 2016). Furthermore, a model checkpoint can be used, where the weights of the trained model at its best accuracy can be saved. Moreover, saving the best model performance as a fine-tuning method can enhance overall performance. Similarly, ensemble learning, like bagging and boosting where predictions from multiple models are combined, can often lead to improved performance and more robust results. Ideally, experimenting with different methods iteratively to refine the model based on empirical results would be the best approach.
<br/>
<br/>
<br/>

### **References** <br/>
1.	Krizhevsky, A., Sutskever, I. and Hinton, G.E. (2017) ‘ImageNet classification with deep convolutional Neural Networks’, Communications of the ACM, 60(6), pp. 84–90.

2.	R, K. (2018) ‘Deep learning for age group Classification System’, International Journal of Advances in Signal and Image Sciences, 4(2), p. 16. 

3.	Kim, J., Lee, J.K. and Lee, K.M., 2016. Accurate image super-resolution using very deep convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1646-1654).

4.	Simonyan, K. and Zisserman, A., 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

