# NYPD Deployment Patterns - Supporting Code
*NOTE*: Reach out for more info about accessing the original dashcam image data. However, we will not be releasing the original dataset due to size constraints (the data clocks in at over 10TB) and privacy considerations. 

# Model Training - Compute Resources 
To train our YOLOv7 object classifier, we utilized the following hardware: 
- 4x RTX A6000 GPU
- 8x CPU (Emma, know specifics?) 
- 256GB RAM 
This allowed us to use a batch size of 40. 
