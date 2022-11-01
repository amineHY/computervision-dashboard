[![Docker image](https://img.shields.io/badge/-Docker%20image-black?logo=docker)](https://hub.docker.com/r/aminehy/computervision_dashboard)



- [:rocket: Computer Vision Dashboard :rocket:](#rocket-computer-vision-dashboard-rocket)
  - [GitHub URL](#github-url)
  - [TODO](#todo)
  - [Features](#features)
  - [Architecture](#architecture)
  - [Demo](#demo)
    - [Video Applications](#video-applications)
    - [Image Applications](#image-applications)
  - [Launch the Dashboard](#launch-the-dashboard)
    - [Run the dashboard from source](#run-the-dashboard-from-source)
      - [Prepare Python virtual environnement](#prepare-python-virtual-environnement)
      - [Run the dashboard](#run-the-dashboard)
    - [Run the dashboard from docker](#run-the-dashboard-from-docker)
      - [(Optional) Build the docker image](#optional-build-the-docker-image)
      - [Run the dashboard](#run-the-dashboard-1)
- [Tools](#tools)

# :rocket: Computer Vision Dashboard :rocket:






---

## GitHub URL

https://github.com/amineHY/computervision-dashboard.git

## TODO

- Redis : think about reducing the API call
- Think about a use case
- Update the GIFs
- reduce duplicated code in the main file

---

## Features

- Apply Deep learning models for Computer Vision
- Support for images and videos
- Loading data from different source: local, web (URL), upload
- It is designer with a modular architecture, where each app is a class
- Export Analytics KPI to a CSV file
- Display Analytics
- Backend and Frontend separated and shipped in their respective docker image
- Librairies: OpenCV, Python, pandas, Tensorflow, Streamlit
  - Frontend developed with python and streamlit
  - Backend developed in python with FastAPI, OpenCV, Tensorflow...

---

## Architecture

Add and image architecture HERE

![](images/2022-11-01-12-51-37.png)

---

## Demo

### Video Applications

- Object Detection
  ![](images/Peek%202022-10-31%2018-44.gif)

- Heatmap Motion Detection
  ![](images/Peek%202022-10-31%2018-38.gif)

---

### Image Applications

- Object detection
  ![](images/2022-10-31-18-29-24.png)
- Face detection
  ![](images/2022-10-31-18-32-05.png)
- Face detection with blurring
  ![](images/2022-10-31-18-33-04.png)

- Face Mask Detection
  ![](images/2022-10-31-18-34-00.png)

---

## Launch the Dashboard

### Run the dashboard from source

#### Prepare Python virtual environnement

- Create a python virtual environnement using `requirements.txt`

  ```
  pipenv install -r requirements.txt
  ```

  Note the path for the created folder `venv_folder`
  ![](images/2022-10-31-17-22-27.png)

- Activate the environnement
  ```
  source venv_folder/bin/activate
  ```
  or
  ```
  pipenv shell
  ```

#### Run the dashboard

- First run this command from the terminal

  ```
  streamlit run main.py
  ```

  ![](images/2022-10-31-17-16-59.png)

- Click on this adresse to open the dashboard on the browser

  ```
  Local URL: http://localhost:8502
  ```

  ![](images/Peek%202022-10-31%2018-52.gif)

---

### Run the dashboard from docker

#### (Optional) Build the docker image

- Build docker image
```
docker build -t aminehy/computervision-dashboard:latest .
```

- Push the docker image to Docker Hub
```
docker login
docker push aminehy/computervision-dashboard:latest
```


#### Run the dashboard

```
docker run -it --rm aminehy/computervision-dashboard:latest streamlit run main.py --server.port 8050
```

![](images/2022-10-31-19-41-52.png)

---


# Tools

- Python
- -Docker


