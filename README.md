This work is a  hybrid  fraud detection system which  integrates real time  interpretability, enabling financial security analysts to gain actionable insights when fraud alerts are triggered. These insights provide a valuable basis for informed decision-making and serve as a starting point for engaging with customers to effectively resolve issues.  

### The repository of this work can be cloned and tested via Github using 

> git clone https://github.com/Ifeanyi-Es/ccfraud_App.git 

### rebuilt locally from the project directory
> cd ccfraud_App

> docker build -t ie-ccfds .

### or rebuilt directly from github using
> docker build -t ie-ccfds https://github.com/Ifeanyi-Es/ccfraud_App.git


### A built image is accessible via docker.
> docker pull ifeanyies/ie-ccfds:latest

### run command
> docker run -d -p 8501:8501 ifeanyies/ie-ccfds:latest

>The application will be available via web browser
http://localhost:8501

### Thank you.
