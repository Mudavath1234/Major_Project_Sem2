# Major_Project_Sem2
This Project is Done in the University for the academic &amp; Individual purpose.

Title : Traffic Monitoring System using Python, OpenCV & YOLOv8

 What This Project Does
This project is a real-time traffic monitoring system  developed using  Python, OpenCV, and YOLOv8 
It performs the following tasks:
	  Detects and classifies multiple types of vehicles in video footage (e.g., cars, bikes, trucks)
    Tracks each vehicle using a centroid-based tracking algorithm
    	Calculates vehicle speed by measuring movement between two Regions of Interest (ROIs)
    	Flags vehicles that violate predefined speed limits
    	Differentiates vehicles moving in forward and backward directions
    	Counts and logs all detected vehicles
    	Exports the results to text files and Excel sheets

Why This Project is Useful
Traditional traffic monitoring systems use radar/lidar, which are expensive and have limited range.  
This AI-powered solution offers a  cost-effective ,  software-only , and  easy-to-deploy method for:
    	Intelligent traffic management
    	Speed violation detection
    	Urban planning
    	Public safety enforcement
It is ideal for deployment at  intersections, highways, toll booths , or  surveillance feeds .

Tech Stack Used & Their Role in the Project
    	Python 3.10: Core programming language used to build the system logic and integrate all modules.         
    	OpenCV (cv2):Handles video processing, frame extraction, drawing bounding boxes, and region overlays.
    	YOLOv8 (Ultralytics) :Deep learning model used for real-time vehicle detection and classification in video.
    	NumPy : Performs numerical operations such as centroid calculations and distance measurement.       
    	Custom Centroid Tracker : Assigns unique IDs to each vehicle and tracks their movement across video frames.
    	Pandas (optional): Can be used to export vehicle data (e.g., speed, ID, time) into structured formats like Excel.


Summary of Tech Stacks:
    	YOLOv8: Detects vehicles in each video frame.
    	OpenCV: Reads video input, displays detections, and defines regions of interest.
    	Centroid Tracker: Tracks the same vehicle across multiple frames to calculate speed and direction.
    	NumPy: Efficiently handles array-based calculations, like centroid positions and Euclidean distances.
IDE & Software Requirements
Recommended IDEs
You can use any Python IDE or code editor. Recommended options:
    	Visual Studio Code (VS Code)
    	Jupyter Notebook (via Anaconda)
    	PyCharm(optional)

Software & Libraries to Install
Ensure Python 3.10+ is installed. Then install the required libraries using **pip**.
Install One-by-One (Beginner Friendly):
    	pip install opencv-python
    	pip install numpy
    	pip install ultralytics
Or Install All at Once:
    	pip install opencv-python numpy ultralytics
Running the Project
    	python test.py
View Output Video
    	python play.py
Output
    	Annotated video saved as output
    	Vehicle data saved to .txt or .xlsx formats (if implemented)
