#Use the official Python image as the base image
#FROM python:3.8-slim

# Set the working directory in the container
#WORKDIR /app

# Copy the current directory contents into the container at /app
#COPY . /app

# Install necessary dependencies
#RUN pip3 install torch torchvision flask pillow opencv-python

# Expose the port on which Flask will run (assuming you are using the default Flask port 5000)
#EXPOSE 5000

# Start the Flask application
#CMD ["python", "app.py"]
# Use the official Python image as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary dependencies for the main Flask app
RUN pip3 install torch torchvision flask pillow opencv-python

# Copy the entire yolov5 folder to /app/yolov5 in the container
COPY yolov5 /app/yolov5

# Install the requirements for the yolov5 folder
WORKDIR /app/yolov5
RUN pip3 install -r requirements.txt

# Expose the port on which Flask will run (assuming you are using the default Flask port 5000)
EXPOSE 7000

# Start the Flask application
CMD ["python", "app.py"]

