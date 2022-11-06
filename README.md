# :rocket: Computer Vision Dashboard :rocket:

---

## Table of content

[![Docker image](https://img.shields.io/badge/-Docker%20image-black?logo=docker)](https://hub.docker.com/repository/docker/aminehy/computervision-dashboard)

- [:rocket: Computer Vision Dashboard :rocket:](#rocket-computer-vision-dashboard-rocket)
  - [Table of content](#table-of-content)
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

---

## GitHub URL

Link to this [GitHub Repo](https://github.com/amineHY/computervision-dashboard.git).

## TODO

- Redis : think about reducing the API call
- Think about a use case
- Update the GIFs
- Create a release for the dashboard
- Add AirFlow

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

![image](images/2022-11-01-12-51-37.png)

---

## Demo

### Video Applications

![image](images/Peek%202022-11-06%2019-48_video_app.gif)

<!-- - Object Detection
  ![image](images/Peek%202022-10-31%2018-44.gif)

- Heatmap Motion Detection
  ![image](images/Peek%202022-10-31%2018-38.gif) -->

---

### Image Applications

![](images/Peek%202022-11-06%2019-46_image_applications.gif)

<!-- - Object detection
  ![image](images/2022-10-31-18-29-24.png)
- Face detection
  ![image](images/2022-10-31-18-32-05.png)
- Face detection with blurring
  ![image](images/2022-10-31-18-33-04.png)

- Face Mask Detection
  ![image](images/2022-10-31-18-34-00.png) -->

---

## Launch the Dashboard

### Run the dashboard from source

#### Prepare Python virtual environnement

- Create a python virtual environnement using `requirements.txt`

  ```shell
  pipenv install -r requirements.txt
  ```

  Note the path for the created folder `venv_folder`
  ![image](images/2022-10-31-17-22-27.png)

- Activate the environnement

  ```shell
  source venv_folder/bin/activate
  ```

  or

  ```shell
  pipenv shell
  ```

#### Run the dashboard

- First run this command from the terminal

  ```shell
  streamlit run main.py
  ```

  ![image](images/2022-10-31-17-16-59.png)

---

### Run the dashboard from docker

#### (Optional) Build the docker image

- Build docker image

```shell
docker build -t aminehy/computervision-dashboard:latest .
```

- Push the docker image to Docker Hub

```shell
docker login
docker push aminehy/computervision-dashboard:latest
```

#### Run the dashboard

```shell
docker run -it --rm aminehy/computervision-dashboard:latest streamlit run main.py --server.port 8050
```

![image](images/2022-10-31-19-41-52.png)
