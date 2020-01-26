# Adversarial-Joint-Distribution-Learning-for-Novel-Class-Sketch-Based-ImageRetrieval
In the information retrieval task, sketch-based image retrieval (SBIR) has drawn significant attention owing to the ease with which sketches can be drawn. The existing deep learning methods for the SBIR are very unrealistic in the real scenario, and its performance reduces drastically for unseen class test examples. Recently, Zero-Shot Sketch-Based Image Retrieval (ZS-SBIR) has drawn a lot of attention due to its ability to retrieve the novel/unseen class images at test time. These methods try to project sketch features into the image domain by learning a distribution conditioned on the sketch. We propose a new framework for ZS-SBIR that models joint distribution between the sketch and image domain using a generative adversarial network. The joint distribution modeling ability of our generative model helps to reduce the domain gap between the sketches and images. Our framework helps to synthesize the novel class image features using sketch features. The generative ability of our model for the unseen/novel classes, conditioned on sketch feature, allows it to perform well on the seen as well as unseen class sketches. We conduct extensive experiments on two widely used SBIR benchmark datasets-Sketchy and Tu-Berlin and obtain significant improvement over the existing state-of-the-art.
## Dataset File Description: 

We have used the ResNet-152 network pre-trained on the ImageNet-1000 dataset to extract features of sketches and images.

###Train set
1. trainData.npy -- contains features of images in the train set of dimensions (104000, 2048) extracted from the ResNet-152 network.

2. trainAttribute.npy -- It is a (104000, 2348) dimension matrix. It has 2048 dimensional features extracted from the ResNet-152 network corresponding to the sketches in the train set concatenated with the 300 dimensions word2vec representation of the class labels of the sketches. However, as you can see in the code, we have not used the word2vec representation of the class labels to train/test the network.

###Test set
1. testData.py -- contains features of images in the test set of dimensions (10453, 2048) extracted from the ResNet-152 network. 

2. testLabel.py -- contains class labels of the images in the testData.py. It is of dimension (10453,1).

3. AlltestAttribute.py -- It is a (12694, 2348) dimension matrix. It has 2048 dimensional features extracted from the ResNet-152 network corresponding to the sketches in the test set concatenated with the 300 dimensions word2vec representation of the class labels of the sketches. However, as you can see in the code, we have not used the word2vec representation of the class labels to train/test the network.

4. AlltestAttribute-label.py -- It contains class labels of sketches in the AlltestAttribute.py file.


For further details, please refer to our paper http://openaccess.thecvf.com/content_ICCVW_2019/papers/MDALC/Pandey_Adversarial_Joint-Distribution_Learning_for_Novel_Class_Sketch-Based_Image_Retrieval_ICCVW_2019_paper.pdf
