# surface-classification-app
Python app that uses an ML model on a live camera feed to describe textures / patterns of surfaces.

Trained ML model on the Describable Textures Dataset (DTD) using PyTorch, used OpenCV to access live camera footage from computer / phone, and PyQt5 to create the GUI.

**Instructions**
1. Run model.py to train the model and save into dtd_resnet50_model.pth
2. Run camera.py to run main program
3. If 'camera' attribute in CameraFeed constructor is equal to 0, use iPhone (or other connected device) as camera, and if it is equal to 1, use computer camera
4. Point camera towards a surface to get the predicted texture / pattern


**Dataset Citation:**
```
@InProceedings{cimpoi14describing,
	      Author    = {M. Cimpoi and S. Maji and I. Kokkinos and S. Mohamed and and A. Vedaldi},
	      Title     = {Describing Textures in the Wild},
	      Booktitle = {Proceedings of the {IEEE} Conf. on Computer Vision and Pattern Recognition ({CVPR})},
	      Year      = {2014}}
```
