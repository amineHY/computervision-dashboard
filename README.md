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
      - [Build the docker image](#build-the-docker-image)
      - [Run the dashboard](#run-the-dashboard-1)

# :rocket: Computer Vision Dashboard :rocket:

---

## GitHub URL

https://github.com/amineHY/computervision-dashboard.git

## TODO

- Redis : Database
- Think about a usecase
- Update the GIFs


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

#### Build the docker image

```
docker build -t computervision_dashboard .
```

#### Run the dashboard

```
docker run -it --rm computervision_dashboard:latest streamlit run main.py --server.port 8050
```

![](images/2022-10-31-19-41-52.png)

---
