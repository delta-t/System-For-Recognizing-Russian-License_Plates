# In our example we want to import the python image from Dockerhub
FROM python:latest

STOPSIGNAL SIGINT
# In order to launch our python code, we must import the application-scripts.
# Here we put files at the image '/alpr/' folder. (automatic license plate recognition system)
COPY . /alpr/

# This command changes the base directory of the image.
# Here we define '/alpr/' as base directory where all commands will be executed
WORKDIR /alpr/

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]