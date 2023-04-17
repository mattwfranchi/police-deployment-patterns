We utilize [Scale](https://scale.com/) for our data annotation requirements in this project. This is a platform very similar to Amazon MTurk, only with quality-of-life improvements for both data seekers and labelers. 
Here, we provide materials that would allow the recreation of our annotation pipeline. 

## Labeling Task Instructions 
Scale.AI requires a detailed, thorough set of instructions for any labeling task. We produced a set of instructions based on a prior paper, *Trucks Don't Mean Trump*, and reviewed labeler feedback prior to launching large batches. Our instruction set received 4/5 stars on average from labelers, with the prime concern (no specification for minimum pixel-area size) being addressed prior to deployment. 

The instructions are as follows: 

### Summary 
The purpose of this task is to identify police presence in dash cam images. These images occur at all times of day, across many parts of a city, but you are always looking for police officers or vehicles. Only about 5-10% of these images contain a police presence, so it is imperative to correctly label the few instances of police vehicles / officers. 

### Workflow 
1. Scan across the vehicles visible in the image, looking for warning lights or a light bar on top of the vehicle, obvious police markings like "NYPD" or colored stripes, etc. Mark any vehicles with the appropriate bounding boxes. Police vehicles will not be yellow. 
2. Scan across the pedestrians visible in the image, looking for uniforms and dark colors, as well as equipment, etc. You may need to zoom in to identify police officers instead of construction workers, etc. Mark any police officers with the appropriate bounding box.
3. Follow standard occlusion rules: if an officer or vehicle is partially hidden by another object, try and draw a bounding box around the entire object, including obstructions. 
4. Please only draw a bounding box if you can reasonably tell the object is police-belonging after a quick glance. Only label as police vehicle if you can easily see (1) blue stripes, (2) "NYPD", (3) "POLICE" text, (4) light / siren strip on roof of vehicle.

### Fully labeled examples 
*Original photos are not included for due to dataset usage restrictions.*
**Police vehicle**. Any type of police vehicle: typically these are labeled with NYPD or Police or similar words.
**Police officer**. Any person wearing a police uniform: typically a safety vest, police hat, etc.

### Common Mistakes (do NOT mark these as police vehicles please!) 
Taxis (2 photos of taxi false positives) 
Buses (4 photos of bus false positives) 
