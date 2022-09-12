# RotNet
Unsupervised representation learning by predicting Image rotation and used the same pretrained model for classification.

## Task 
Used ResNet50 architecture and changed the lastlayers for Rotation angle classification(0,90,270,360)

-Steps:
 -- Split the dataset in Train,Test,Validation (Path: 'RotNet-master/datasets/load_train_test_val.py')
 -- Trained the model (Path: 'RotNet-master/train/train_street_view.py')
 -- Test the model (Path: 'RotNet-master/test/street_view_test.py')

## Task 2 : 
Used pretrained model and changed the last layers for flower classification.

-Steps:
 -- Use the pretrained model rot_net for image classification
 -- Train the model (Path: classification_model/classification.py)

## Lack of Resources
Due to lack of resources not able to complete train the models.
