import base64
import os
import urllib
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image

import cv2 as cv
import pafy
import streamlit as st
import youtube_dl

##########################################################


class GUI():
    """
    This class is dedicated to manage to user interface of the website. It contains methods to edit the sidebar for the selected application as well as the front page.
    """

    def __init__(self):

        self.list_of_apps = [
            'Empty',
            'Object Detection',
            'Fire Detection',
            'Face Detection_with_Blurring',
            'Face Detection',
            # 'Cars Counting'
        ]

        self.guiParam = {}

    # ----------------------------------------------------------------

    def getGuiParameters(self):
        self.common_config()
        self.appDescription()
        return self.guiParam

        # ------------------------------------a----------------------------

    def common_config(self, title='Computer Vision Dashboard '):  # (Beta version :golf:)
        """
        User Interface Management: Sidebar
        """

        st.title(title)

        st.sidebar.markdown("### Settings")

        # Get the application type from the GUI
        self.appType = st.sidebar.radio(
            ' Image or Video Application?', ['Image Application', 'Video Application'])

        # if self.appType == 'Image Application':
        self.dataSource = st.sidebar.radio(
            'Load data from', ['Database', 'URL', 'Upload'])

        # elif (self.appType == 'Video Application'):
        #     self.dataSource = st.sidebar.radio(
        #         'Load data', ['Database', 'URL', 'Upload'])

        # else:
        #     raise ValueError(
        #         "Please select either 'Image Application' or 'Video Applications'")

        # Get the application from the GUI
        self.selectedApp = st.sidebar.selectbox(
            'Chose a Computer Vision Application', self.list_of_apps)

        if self.selectedApp == 'Empty':
            st.sidebar.warning('Select an application from the list')

        if self.appType == 'Video Application':
            self.recordOutputVideo = st.sidebar.checkbox(
                'Record Video with Overlay', value=True)

            self.frameMax = st.sidebar.slider(
                'Number of frame to process', value=10, min_value=1, max_value=100, step=1)
        else:
            self.recordOutputVideo = False
            self.frameMax = 1

        # self.displayFlag = st.sidebar.checkbox(
        #     'Display Real-Time Results', value=True)

        # Update the dictionnary
        self.guiParam.update(
            dict(selectedApp=self.selectedApp,
                 appType=self.appType,
                 dataSource=self.dataSource,
                 recordOutputVideo=self.recordOutputVideo))

        # --------------------------------------------------------------------------

    def appDescription(self):

        st.header(' :arrow_right: Application: {}'.format(self.selectedApp))

        if self.selectedApp == 'Object Detection':
            st.info(
                'This application performs object detection using advanced deep learning models. It can detects more than 80 object from COCO dataset.')
            self.sidebarObjectDetection()

        elif self.selectedApp == 'Face Detection':
            st.info(
                "This application performs face detection using advanced deep learning models. It can detects face in the image.")
            self.sidebarFaceDetection()

        elif self.selectedApp == 'Face Detection_with_Blurring':
            st.info(
                "This application performs face detection using advanced deep learning models. It can detects face in the image. In addition, to preserve privacy, it blur the detected faces to comply with the RGPD.")
            self.sidebarFaceDetectionWithBlur()

        elif self.selectedApp == 'Fire Detection':
            st.info(
                'This application performs fire detection using advanced deep learning models. ')
            self.sidebarFireDetection()

        elif self.selectedApp == 'Cars Counting':
            st.info(
                'This application performs object counting using advanced deep learning models. It can detects more than 80 object from COCO dataset.')
            self.sidebarCarsCounting()

        else:
            st.info(
                'To start using InVeesion dashboard you must first select an Application from the sidebar menu other than Empty')

    # --------------------------------------------------------------------------
    def sidebarEmpty(self):
        pass
    # --------------------------------------------------------------------------

    def sidebarFaceDetection(self):
        """
        """

        # st.sidebar.markdown("### :arrow_right: Model")
        # --------------------------------------------------------------------------
        model = st.sidebar.selectbox(
            label='Select the model',
            options=(['MobileNetSSD']))

        st.sidebar.markdown("### :arrow_right: Parameters")
        # --------------------------------------------------------------------------
        confThresh = st.sidebar.slider(
            'Confidence', value=0.40, min_value=0.0, max_value=1.00, step=0.05)
        faceBlur = False

        self.guiParam.update(dict(confThresh=confThresh,
                                  model=model,
                                  faceBlur=faceBlur))

    # --------------------------------------------------------------------------

    def sidebarFaceDetectionWithBlur(self):
        """
        """

        # st.sidebar.markdown("### :arrow_right: Model")
        # --------------------------------------------------------------------------
        model = st.sidebar.selectbox(
            label='Select the model',
            options=(["MobileNetSSD"]))

        st.sidebar.markdown("### :arrow_right: Parameters")
        # --------------------------------------------------------------------------
        confThresh = st.sidebar.slider(
            'Confidence', value=0.40, min_value=0.0, max_value=1.00, step=0.05)
        faceBlur = True
        self.guiParam.update(dict(faceBlur=faceBlur,
                                  confThresh=confThresh,
                                  model=model))

    # --------------------------------------------------------------------------

    def sidebarObjectDetection(self):

        # st.sidebar.markdown("### :arrow_right: Model")
        #------------------------------------------------------#
        model = st.sidebar.selectbox(
            label='Select the model',
            options=['Caffe-MobileNetSSD', 'Darknet-YOLOv3-tiny', 'Darknet-YOLOv3'])

        # st.sidebar.markdown("### Target Object")
        # #------------------------------------------------------#
        # desired_object = st.sidebar.multiselect(
        #     label='What object would like to detect?',
        #     options=('person', 'cars', 'cell phone', 'plane', 'fire'))

        # st.sidebar.markdown("### :arrow_right: Model Parameters")
        #------------------------------------------------------#
        confThresh = st.sidebar.slider(
            'Confidence', value=0.5, min_value=0.0, max_value=1.0)
        nmsThresh = st.sidebar.slider(
            'Non-maximum suppression', value=0.30, min_value=0.0, max_value=1.00, step=0.05)

        self.guiParam.update(dict(confThresh=confThresh,
                                  nmsThresh=nmsThresh,
                                  model=model,
                                  #   desired_object=desired_object
                                  ))

    # --------------------------------------------------------------------------

    def sidebarFireDetection(self):

        # st.sidebar.markdown("### :arrow_right: Model")
        #------------------------------------------------------#
        model = st.sidebar.selectbox(
            label='Select the model',
            options=['Darknet-YOLOv3-tiny'])

        # st.sidebar.markdown("### :arrow_right: Model Parameters")
        #------------------------------------------------------#
        confThresh = st.sidebar.slider(
            'Confidence', value=0.5, min_value=0.0, max_value=1.0)
        nmsThresh = st.sidebar.slider(
            'Non-maximum suppression', value=0.30, min_value=0.0, max_value=1.00, step=0.05)

        self.guiParam.update(dict(confThresh=confThresh,
                                  nmsThresh=nmsThresh,
                                  model=model))
    # --------------------------------------------------------------------------

    def sidebarCarsCounting(self):

        # st.sidebar.markdown("### :arrow_right: Model")
        #------------------------------------------------------#
        model = st.sidebar.selectbox(
            label='Select the model',
            options=('Model 1', 'Model 2', 'Model 3'))

        self.guiParam.update(dict(model=model))

##########################################################


class DataManager:
    """
    """

    def __init__(self, guiParam):
        self.guiParam = guiParam

        self.url_demo_videos = {
            "Driving car in a city": 'https://www.youtube.com/watch?v=7BjNbkONCFw',
            "A Sample Video with Faces": "https://www.youtube.com/watch?v=ohmajJTcpNk"}

        self.url_demo_images = {
            "NY-City": "https://s4.thingpic.com/images/8a/Qcc4eLESvtjiGswmQRQ8ynCM.jpeg",
            "Paris-street": "https://www.discoverwalks.com/blog/wp-content/uploads/2018/08/best-streets-in-paris.jpg"}

        self.demo_video_examples = {"Street-CCTV": guiParam["path_database"] + "object.mp4",
                                    "Fire on the road": guiParam["path_database"] + "fire.mp4"}

        self.demo_image_examples = {"Family-picture": guiParam["path_database"] + "family.jpg",
                                    "Fire": guiParam["path_database"] + "fire.jpg",
                                    "Dog": guiParam["path_database"] + "dog.jpg",
                                    "Crosswalk": guiParam["path_database"] + "demo.jpg",
                                    "Cat": guiParam["path_database"] + "cat.jpg",
                                    "Car on fire": guiParam["path_database"] + "car_on_fire.jpg"}

        self.image = None
        self.image_byte = None

        self.video = None
        self.video_byte = None
        self.data = None
        self.data_byte = None

  #################################################################
  #################################################################

    def load_image_source(self):
        """
        """

        if self.guiParam["dataSource"] == 'Database':

            # @st.cache(allow_output_mutation=True)
            def load_image_from_path(image_path):
                # im_rgb = cv.imread(image_path, cv.IMREAD_COLOR)
                im_byte = open(image_path, 'rb').read()
                im_ndarr = np.frombuffer(im_byte, dtype=np.uint8)
                im_rgb = cv.imdecode(im_ndarr, cv.IMREAD_COLOR)
                return im_rgb, im_byte

            file_path = st.text_input('Enter the image PATH')

            if os.path.isfile(file_path):
                self.image, self.image_byte = load_image_from_path(
                    image_path=file_path)

            elif file_path == "":
                file_path_idx = st.selectbox(
                    'Or select a demo image from the list', list(self.demo_image_examples.keys()))
                file_path = self.demo_image_examples[file_path_idx]

                self.image, self.image_byte = load_image_from_path(
                    image_path=file_path)
            else:
                raise ValueError("[Error] Please enter a valid image path")

            #--------------------------------------------#
            #--------------------------------------------#

        elif self.guiParam["dataSource"] == 'Upload':

            @st.cache(allow_output_mutation=True)
            def load_image_from_upload(file):
                im_byte = file.read()
                im_ndarr = np.frombuffer(im_byte, dtype=np.uint8)
                im_rgb = cv.imdecode(im_ndarr, cv.IMREAD_COLOR)
                return im_rgb, im_byte

            file_path = st.file_uploader(
                'Upload an image', type=['png', 'jpg'])

            if file_path != None:
                self.image, self.image_byte = load_image_from_upload(file_path)
            elif file_path == None:
                raise ValueError(
                    "[Error] Please upload a valid image ('png', 'jpg')")
            #--------------------------------------------#
            #--------------------------------------------#

        elif self.guiParam["dataSource"] == 'URL':

            @st.cache(allow_output_mutation=True)
            def load_image_from_url(url_image):
                file = urllib.request.urlopen(url_image)
                im_byte = file.read()
                # tmp = np.asarray(bytearray(file.read()), dtype=np.uint8)
                im_ndarr = np.frombuffer(im_byte, dtype=np.uint8)
                im_rgb = cv.imdecode(im_ndarr, cv.IMREAD_COLOR)
                return im_rgb, im_byte

            file_path = st.text_input('Enter the image URL')

            if file_path != "":
                self.image, self.image_byte = load_image_from_url(
                    url_image=file_path)

            elif file_path == "":

                file_path_idx = st.selectbox(
                    'Or select a URL from the list', list(self.url_demo_images.keys()))
                file_path = self.url_demo_images[file_path_idx]

                self.image, self.image_byte = load_image_from_url(
                    url_image=file_path)
            else:
                raise ValueError("[Error] Please enter a valid image URL")

            #--------------------------------------------#
            #--------------------------------------------#

        else:
            raise ValueError("Please select one source from the list")

        return self.image, self.image_byte

    def load_image_or_video(self):
        """
        Handle the data input from the user parameters
        """
        if self.guiParam['appType'] == 'Image Application':
            self.data, self.data_byte = self.load_image_source()

        elif self.guiParam['appType'] == 'Video Application':
            self.data, self.data_byte = self.load_video_source()

        else:
            raise ValueError(
                '[Error] Please select of the two Application pipelines')

        return self.data, self.data_byte

    def check_video_object(self, video):
        """
        """
        if video != None and video.isOpened():
            print('[INFO] File is correctly openned')

        else:
            raise Exception(
                '[Error] Could not open the video source: {}'.format(self.guiParam['dataSource']))

    def load_video_source(self):
        """
        """
        # print(self.guiParam["dataSource"])
        if self.guiParam["dataSource"] == "None":
            self.video = None
            st.warning("No application is selected.")

            #------------------------------------------------------#
            #------------------------------------------------------#
        elif self.guiParam["dataSource"] == 'Upload':
            def load_video_from_upload(file_path):
                return None, file_path

            file_path = st.file_uploader(
                "Upload a video (200 Mo maximum) ...", type=["mp4", "mpeg", 'avi'])
            print(file_path)
            if file_path != None:
                self.video, self.video_byte = load_video_from_upload(file_path)
            elif file_path == None:
                raise ValueError(
                    "[Error] Please upload a valid image ('mp4', 'mpeg', 'avi')")

            #------------------------------------------------------#
            #------------------------------------------------------#
        elif self.guiParam["dataSource"] == 'Database':

            @st.cache(allow_output_mutation=True)
            def load_video_from_path(video_path):
                isinstance(video_path, str)
                video = cv.VideoCapture(video_path)
                video_byte = open(video_path, "rb")
                return video, video_byte

            file_path = st.text_input(
                'Enter PATH of the video')

            if os.path.isfile(file_path):
                self.video, self.video_byte = load_video_from_path(file_path)
            elif file_path == "":
                file_path_idx = st.selectbox(
                    'Or select a demo image from the list', list(self.demo_video_examples.keys()))
                file_path = self.demo_video_examples[file_path_idx]
                self.video, self.video_byte = load_video_from_path(
                    video_path=file_path)

            else:
                raise ValueError("[Error] Please enter a valid video path")

            #------------------------------------------------------#
            #------------------------------------------------------#

        # elif self.guiParam["dataSource"] == 'Webcam':

        #     @st.cache(allow_output_mutation=True)
        #     def load_video_from_webcam(webcam_id):
        #         isinstance(webcam_id, int)
        #         video = cv.VideoCapture(webcam_id)
        #         from time import time
        #         time.sleep(2)
        #         return video

        #     webcam_id = 0  # default value, change it to the suitable device
        #     self.video = load_video_from_webcam(webcam_id)

            #------------------------------------------------------#
            #------------------------------------------------------#

        elif self.guiParam["dataSource"] == 'URL':
            def url_to_image(url):
                # download the image, convert it to a NumPy array, and then read
                # it into OpenCV format
                resp = urllib.urlopen(url)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv.imdecode(image, cv.IMREAD_COLOR)
                # return the image
                return image

            @st.cache(allow_output_mutation=True)
            def load_video_from_url(video_url):
                isinstance(video_url, str)
                video = pafy.new(video_url)
                videoHightRes = video.getbest(preftype="mp4")
                videoHightRes.download('demo.mp4')
                video_byte = open('demo.mp4', 'rb')

                return video, video_byte

            video_url = st.text_input('Enter URL of the video')
            st.info(
                'Samples here: https://research.google.com/youtube8m/explore.html')

            if video_url != "":
                self.video, self.video_byte = load_video_from_url(video_url)

            elif video_url == "":
                video_url_idx = st.selectbox(
                    'Or select a URL from the list', list(self.url_demo_videos.keys()))
                video_url = self.url_demo_videos[video_url_idx]
                self.video, self.video_byte = load_video_from_url(video_url)

            else:
                raise ValueError("[Error] Please enter a valid video URL")

            #------------------------------------------------------#

        else:
            raise ValueError("Please select a 'Data Source' from the list")

        # self.check_video_object(self.video)

        return self.video, self.video_byte

##########################################################


def main():
    """
    """

    paths = {
        "path_database": "database/",
        "path_results": "app/results/",
        "path_model": "app/models/"
    }

    guiParam = GUI().getGuiParameters()
    guiParam.update(paths)

    # Send Get Request
    if guiParam["selectedApp"] != 'Empty':

        print("\n[INFO] Sending Get Request ...")

        if guiParam['appType'] == 'Image Application':
            __, image_byte = DataManager(guiParam).load_image_or_video()
            fastapi_post_url = "http://127.0.0.1:8000/image/"
            response = requests.post(fastapi_post_url,
                                     params=guiParam,
                                     files={"image": base64.b64encode(image_byte)})

        elif guiParam['appType'] == 'Video Application':
            video, video_byte = DataManager(guiParam).load_image_or_video()

            # print(type(video))
            # print(type(video_byte))

            # ff = "/media/amine/DATA/4K Road traffic video for object detection and tracking - free download now!-MNn9qKG2UFI.webm"
            # video_byte = open(ff, 'rb')

            fastapi_post_url = "http://127.0.0.1:8000/video/"
            response = requests.post(fastapi_post_url,
                                     params=guiParam,
                                     files={"video": video_byte})
        print(response.url)

        if response:
            print('\nRequest is successful.')
            # print("\n[INFO] API Response: ", response.json())

            st.markdown("## Results")
            res_json = response.json()['response']
            keys = list(res_json.keys())
            values = list(res_json.values())

            # display data in the frontend
            if response.json()["media"] == "video":

                received_data = "data_from_api/"
                with open(received_data+'get_demo.mp4', 'wb') as vid_byte:
                    vid_byte.write(base64.b64decode(values[0]))
                with open(received_data+'get_demo.csv', 'wb') as csv_byte:
                    csv_byte.write(base64.b64decode(values[1]))

                st.dataframe(pd.read_csv(received_data+'get_demo.csv'))
                st.video(open(received_data+'get_demo.mp4', 'rb').read())

            elif response.json()["media"] == "image":
                os.system('pwd')
                # parse response and extract data (image + csv)
                received_data = "data_from_api/"
                with open(received_data+'get_demo.png', 'wb') as im_byte:
                    im_byte.write(base64.b64decode(values[0]))

                with open(received_data+'get_demo.csv', 'wb') as csv_byte:
                    csv_byte.write(base64.b64decode(values[1]))

                st.dataframe(pd.read_csv(received_data+'get_demo.csv'))

                st.image(open(received_data+'get_demo.png', 'rb').read(),
                         channels="BGR",  use_column_width=True)

        else:
            print('\nRequest returned an error.')

    else:
        st.warning("Please select an application")


if __name__ == "__main__":
    main()
