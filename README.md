# :rocket: Computer Vision Dashboard :rocket:

## Create and activate Python virtual env

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


## Launch the Dashboard
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



## Demo

### Video
 - Object Detection
  ![](images/Peek%202022-10-31%2018-44.gif)
  
 - Heatmap Motion Detection
  ![](images/Peek%202022-10-31%2018-38.gif)


### Image Applications
- Object detection
  ![](images/2022-10-31-18-29-24.png)
- Face detection
  ![](images/2022-10-31-18-32-05.png)
- Face detection with blurring
  ![](images/2022-10-31-18-33-04.png)

- Face Mask Detection
 ![](images/2022-10-31-18-34-00.png)


