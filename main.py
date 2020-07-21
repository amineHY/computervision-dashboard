#----------------------------------------------------------------#
"""
Author: Amine Hadj-Youcef
Email: hadjyoucef.amine@gmail.com
Website: www.amine-hy.com
Github: https://github.com/amineHY
"""
#----------------------------------------------------------------#
import base64
import os
import urllib
from collections import Counter
from io import BytesIO

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import youtube_dl
from matplotlib import pyplot as plt
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots

print(__doc__)

#----------------------------------------------------------------#
#----------------------------------------------------------------#


class GUI():
    """
    This class is dedicated to manage to user interface of the website. It contains methods to edit the sidebar for the selected application as well as the front page.
    """

    def __init__(self):

        self.list_of_apps = [
            'Empty',
            "Face Mask Detection",
            'Face Detection with Blurring',
            'Face Detection',
            'Object Detection',
            'Fire Detection',
            'Heatmap Motion',
        ]

        self.guiParam = {}

    # ----------------------------------------------------------------

    def getGuiParameters(self):
        self.common_config()
        self.appDescription()
        return self.guiParam

        # ------------------------------------a----------------------------

    def common_config(self, title='InVeesion Dashboard 🚀'):  # (Beta version :golf:)
        """
        User Interface Management: Sidebar
        """

        st.title(title)

        st.sidebar.title("Settings")

        # Get the application type from the GUI
        self.appType = st.sidebar.radio(
            ' Image or Video Application?', ['Image Application', 'Video Application'])

        # if self.appType == 'Image Application':
        self.dataSource = st.sidebar.radio(
            'Load data from', ['Database', 'URL', 'Upload'])

        if self.appType == 'Video Application':
            self.recordOutputVideo = st.sidebar.checkbox(
                'Record Video with Overlay', value=True)

            self.frameFreq = st.sidebar.slider(
                'Frame Frequency', value=15, min_value=1, max_value=60, step=1)

            self.frameMax = st.sidebar.slider(
                'Frames to process', value=100, min_value=self.frameFreq, max_value=500, step=1)

        elif self.appType == 'Image Application':
            self.recordOutputVideo = False
            self.frameMax = 1
            self.frameFreq = 1

        # Get the application from the GUI
        self.selectedApp = st.sidebar.selectbox(
            'Chose a Computer Vision Application', self.list_of_apps)

        if self.selectedApp == 'Empty':
            st.sidebar.warning('Select an application from the list')

        # Update the dictionnary
        self.guiParam.update(
            dict(selectedApp=self.selectedApp,
                 appType=self.appType,
                 dataSource=self.dataSource,
                 recordOutputVideo=self.recordOutputVideo,
                 frameMax=self.frameMax,
                 frameFreq=self.frameFreq))

        # --------------------------------------------------------------------------

    def appDescription(self):

        st.header('{}'.format(self.selectedApp))

        if self.selectedApp == 'Object Detection':
            st.info(
                'This application performs object detection using advanced deep learning models. It can detects more than 80 object from COCO dataset.')
            self.sidebarObjectDetection()

        elif self.selectedApp == 'Face Detection':
            st.info(
                "This application performs face detection using advanced deep learning models. It can detects face in the image.")
            self.sidebarFaceDetection()

        elif self.selectedApp == 'Face Detection with Blurring':
            st.info(
                "This application performs face detection using advanced deep learning models. It can detects face in the image. In addition, to preserve privacy, it blur the detected faces to comply with the RGPD.")
            self.sidebarFaceDetectionWithBlur()

        elif self.selectedApp == 'Fire Detection':
            st.info(
                'This application performs fire detection using advanced deep learning models. ')
            self.sidebarFireDetection()

        elif self.selectedApp == "Face Mask Detection":
            st.info(
                'This application performs Face Mask Detection')
            self.sidebarFaceMaskDetection()

        elif self.selectedApp == 'Heatmap Motion':
            st.info(
                'This application performs heatmap motion. It detect part of the video where there a concentrated movement.')
            self.sidebarHeatmapMotion()

        else:
            st.info(
                'To start using InVeesion dashboard you must first select an Application from the sidebar menu other than Empty')

    # --------------------------------------------------------------------------
    def sidebarEmpty(self):
        pass
    # --------------------------------------------------------------------------

    def sidebarHeatmapMotion(self):

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
            'Confidence', value=0.60, min_value=0.0, max_value=1.00, step=0.05)

        self.guiParam.update(dict(confThresh=confThresh,
                                  model=model))

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
            'Confidence', value=0.60, min_value=0.0, max_value=1.00, step=0.05)
        self.guiParam.update(dict(
            confThresh=confThresh,
            model=model))
    # --------------------------------------------------------------------------

    def sidebarFaceMaskDetection(self):
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
            'Confidence', value=0.60, min_value=0.0, max_value=1.00, step=0.05)

        self.guiParam.update(dict(
            confThresh=confThresh,
            model=model))

    # --------------------------------------------------------------------------

    def sidebarObjectDetection(self):

        # st.sidebar.markdown("### :arrow_right: Model")
        #------------------------------------------------------#
        model = st.sidebar.selectbox(
            label='Select the model',
            options=['Caffe-MobileNetSSD', 'Darknet-YOLOv3-tiny', 'Darknet-YOLOv3'])

        st.sidebar.markdown("### Object Filtering")
        #------------------------------------------------------#
        allowedLabel = st.sidebar.multiselect(
            label='What object would like to detect?',
            options=('person', 'car', 'bicycle', 'dog', 'cell phone', 'plane', 'fire'))

        allowedLabel = ['all'] if len(allowedLabel) == 0 else allowedLabel

        st.sidebar.markdown("### :arrow_right: Model Parameters")
        #------------------------------------------------------#
        confThresh = st.sidebar.slider(
            'Confidence', value=0.6, min_value=0.0, max_value=1.0)
        nmsThresh = st.sidebar.slider(
            'Non-maximum suppression', value=0.30, min_value=0.0, max_value=1.00, step=0.05)

        self.guiParam.update(dict(confThresh=confThresh,
                                  nmsThresh=nmsThresh,
                                  model=model,
                                  allowedLabel=allowedLabel
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
            'Confidence', value=0.6, min_value=0.0, max_value=1.0)
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

#----------------------------------------------------------------#


def hide_streamlit_footer():
    hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """
    st.markdown(hide_footer_style, unsafe_allow_html=True)

#----------------------------------------------------------------#
def postprocessing_object_detection_df(df):
    df_ = df.copy()

    # unwrap bboxes
    df_.bboxes = df.bboxes.apply(lambda x: pd.eval(x))
    df_.confidences = df.confidences.apply(lambda x: pd.eval(x))
    df_.predClasses = df.predClasses.apply(lambda x: pd.eval(x))

    if "predClasses" in df_.columns:
        df_.loc[:, 'counting_obj'] = df_["predClasses"].apply(
            lambda x: Counter(x)).values
        df_.loc[:, 'objectClass'] = df_.loc[:, 'counting_obj'].apply(lambda x: list(x.keys())).values
        df_.loc[:, 'objectNumb'] = df_.loc[:, 'counting_obj'].apply(
            lambda x: list(x.values())).values
        
        df_classes = pd.DataFrame(df_.counting_obj.to_dict()).T
        dataf = df_.join(df_classes)

    return dataf, df_classes


def disp_analytics(df, df_classes):
    if len(df_classes.columns)>0:
        st.markdown("## Global Analytics")
        fig = px.bar(x=df_classes.columns, y=df_classes.sum())
        fig.update_layout(height=400, width=900, title_text="...", yaxis=dict(
            title_text="Number of Detection"), xaxis=dict(title_text="Detection Object in the Video"))
        st.plotly_chart(fig)

        fig = px.pie(df_classes, values=df_classes.sum(),
                    names=df_classes.columns, title='Detected Objects in the Video')
        st.plotly_chart(fig)

        fig = make_subplots(rows=len(df_classes.columns), cols=1)
        for idx, feat in enumerate(df_classes.columns):
            fig.add_trace(
                go.Scatter(x=df.frameIdx,
                        y=df[feat], mode='lines+markers', name=feat),
                row=idx+1, col=1)
        tmp = (len(df_classes.columns))*400
        fig.update_layout(height=tmp, width=900, title_text="Objects Filtering")
        st.plotly_chart(fig)

        fig = px.scatter(x=df.frameIdx, y=df.total_object)
        fig.update_layout(height=400, width=900,
                        title_text="Total Detection per frame")
        st.plotly_chart(fig)

        st.markdown("## Motion Analytics")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=df.frameIdx,
                                y=df.motion_status, mode='lines'), row=1, col=1)
        fig.update_layout(height=600, width=900, title_text="Detected Motion in the Video", yaxis=dict(
            title_text="Motion Status"), xaxis=dict(title_text="Timestamp"))
        st.plotly_chart(fig)

#----------------------------------------------------------------#

class DataManager:
    def __init__(self, guiParam):
        self.guiParam = guiParam

        self.url_demo_videos = {
            "Driving car in a city": 'https://www.youtube.com/watch?v=7BjNbkONCFw',
            "A Sample Video with Faces": "https://www.youtube.com/watch?v=ohmajJTcpNk"}

        self.url_demo_images = {
            "NY-City": "https://s4.thingpic.com/images/8a/Qcc4eLESvtjiGswmQRQ8ynCM.jpeg",
            "Paris-street": "https://www.discoverwalks.com/blog/wp-content/uploads/2018/08/best-streets-in-paris.jpg"}

        self.demo_video_examples = {"Street-CCTV": guiParam["path_database"] + "object.mp4",
                                    "Showroom": guiParam["path_database"] + 'showroom.mov'}
        self.demo_image_examples = {"COVID-19 Mask": guiParam["path_database"] + "face_mask.jpeg",
                                    "Family-picture": guiParam["path_database"] + "family.jpg",
                                    "Dog": guiParam["path_database"] + "dog.jpg",
                                    "Crosswalk": guiParam["path_database"] + "demo.jpg",
                                    "Car on fire": guiParam["path_database"] + "car_on_fire.jpg"}

        self.image = None
        self.image_byte = None

        self.video = None
        self.video_byte = None
        self.data = None
        self.data_byte = None

    #--------------------------------------------------------#
    #--------------------------------------------------------#

    def get_video_path(self):

        if self.guiParam["dataSource"] == 'Upload':
            filelike = st.file_uploader(
                "Upload a video (200 Mo maximum) ...", type=["mp4", "mpeg", 'avi'])

            video_path = 'database/uploaded_video.png'

            if filelike:
                with open(video_path, 'wb') as f:
                    f.write(filelike.read())
                return video_path
            else:
                return None

            #------------------------------------------------------#

        elif self.guiParam["dataSource"] == 'Database':

            video_path = st.text_input('Enter PATH of the video')

            if os.path.isfile(video_path):
                return video_path

            else:
                video_path_idx = st.selectbox(
                    'Or select a demo video from the list', list(self.demo_video_examples.keys()))
                video_path = self.demo_video_examples[video_path_idx]
                return video_path

            #------------------------------------------------------#

        elif self.guiParam["dataSource"] == 'URL':

            video_url = st.text_input('Enter URL of the video')
            video_path = 'database/downloaded_video.mp4'

            if video_url != "":
                isinstance(video_url, str)
                print('Downloading ', video_url)
                ydl_opts = dict(
                    format='bestvideo[height<=480]', outtmpl=video_path)
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                return video_path

            else:
                st.info("Here are some video samples" +
                        "\n Driving car in a city: https://www.youtube.com/watch?v=7BjNbkONCFw \
                \n A Sample Video with Faces: https://www.youtube.com/watch?v=ohmajJTcpNk")
                video_url = 'https://www.youtube.com/watch?v=7BjNbkONCFw'

            #------------------------------------------------------#

        else:
            raise ValueError("Please select a 'Data Source' from the list")

    def get_image_path(self):
        #--------------------------------------------#
        if self.guiParam["dataSource"] == 'Database':
            image_path = st.text_input('Enter the image PATH')

            if image_path == '':
                image_path_idx = st.selectbox(
                    'Or select a demo image from the list', list(self.demo_image_examples.keys()))
                image_path = self.demo_image_examples[image_path_idx]
            return image_path

        #--------------------------------------------#

        elif self.guiParam["dataSource"] == 'Upload':
            filelike = st.file_uploader('Upload an image', type=['png', 'jpg'])
            image_path = 'database/uploaded_image.png'

            if filelike != None:
                with open(image_path, 'wb') as f:
                    f.write(filelike.read())
                return image_path
            else:
                raise ValueError('Please Upload an image first')

        #--------------------------------------------#

        elif self.guiParam["dataSource"] == 'URL':
            url_image = st.text_input('Enter the image URL')
            image_path = 'database/downloaded_image.png'

            if url_image != "":
                with open(image_path, 'wb') as f:
                    f.write(urllib.request.urlopen(url_image).read())
                    return image_path
            else:
                url_image_idx = st.selectbox(
                    'Or select a URL from the list', list(self.url_demo_images.keys()))
                url_image = self.url_demo_images[url_image_idx]

    def load_image_source(self):
        """
        """

        if self.guiParam["dataSource"] == 'Database':

            @st.cache(allow_output_mutation=True)
            def load_image_from_path(image_path):
                # im_rgb = cv.imread(image_path, cv.IMREAD_COLOR)
                with open(image_path, 'rb') as f:
                    im_byte = f.read()
                im_ndarr = np.frombuffer(im_byte, dtype=np.uint8)
                im_rgb = cv.imdecode(im_ndarr, cv.IMREAD_COLOR)
                return im_rgb, filelike

                image_path

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
                filelike = file
                # im_ndarr = np.frombuffer(im_byte, dtype=np.uint8)
                # im_rgb = cv.imdecode(im_ndarr, cv.IMREAD_COLOR)
                im_rgb = []
                return im_rgb, filelike

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
                print('Downloading ...')
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
            @st.cache(allow_output_mutation=True)
            def load_video_from_upload(file_path):
                return None, file_path

            file_path = st.file_uploader(
                "Upload a video (200 Mo maximum) ...", type=["mp4", "mpeg", 'avi'])

            if file_path != "":
                self.video, self.video_byte = load_video_from_upload(file_path)
            else:
                st.info("Please upload a valid image ('mp4', 'mpeg', 'avi')")

            # if video_url != "":
            #     self.video, self.video_byte = load_video_from_url(video_url)
            # else:
            #     st.info("Here are some video samples"+
            #     "\n Driving car in a city: https://www.youtube.com/watch?v=7BjNbkONCFw \
            #     \n A Sample Video with Faces: https://www.youtube.com/watch?v=ohmajJTcpNk"
            #     )
            #------------------------------------------------------#
            #------------------------------------------------------#

        elif self.guiParam["dataSource"] == 'Database':

            @st.cache(allow_output_mutation=True)
            def load_video_from_path(video_path):
                isinstance(video_path, str)
                video = cv.VideoCapture(video_path)
                with open(video_path, "rb") as f:
                    video_byte = f.read()
                    # print(type(video_byte))
                return video_path, video_byte

            file_path = st.text_input('Enter PATH of the video')

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

            @st.cache(allow_output_mutation=True)
            def load_video_from_url(video_url):
                isinstance(video_url, str)
                print('Downloading ', video_url)

                ydl_opts = {

                    'format': 'bestvideo[height<=480]',
                    'outtmpl': 'database/tmp.mp4'
                }
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                print(ydl)
                # video = pafy.new(video_url)
                # videoHightRes = video.getbest(preftype="mp4")
                # videoHightRes.download('database/demso.mp4')
                with open('database/tmp.mp4', 'rb') as f:
                    video_byte = f.read()
                    print('reading video')
                os.system("rm database/tmp.mp4")

                video = None
                return video, video_byte

            video_url = st.text_input('Enter URL of the video')
            # st.info(
            #     'Samples here: https://research.google.com/youtube8m/explore.html')

            if video_url != "":
                self.video, self.video_byte = load_video_from_url(video_url)
            else:
                st.info("Here are some video samples" +
                        "\n Driving car in a city: https://www.youtube.com/watch?v=7BjNbkONCFw \
                \n A Sample Video with Faces: https://www.youtube.com/watch?v=ohmajJTcpNk"
                        )
            # elif video_url == "":
            #     video_url_idx = st.selectbox(
            #         'Or select a URL from the list', list(self.url_demo_videos.keys()))
            #     video_url = self.url_demo_videos[video_url_idx]
            #     self.video, self.video_byte = load_video_from_url(video_url)

            # else:
            #     raise ValueError("[Error] Please enter a valid video URL")

            #------------------------------------------------------#

        else:
            raise ValueError("Please select a 'Data Source' from the list")

        # self.check_video_object(self.video)

        return self.video, self.video_byte

#----------------------------------------------------------------#


def main():

    # get parameter for the api
    guiParam = GUI().getGuiParameters()
    api_param = guiParam.copy()

    # define paths
    paths = {
        "path_database": "database/",
        "received_data": "data_from_api/",
    }

    guiParam.update(paths)

    #----------------------------------------------------------------#
    # Send Request to inveesion-API
    #----------------------------------------------------------------#

    if guiParam["selectedApp"] != 'Empty':

        #----------------------------------------------------------------#
        # url_base="https://inveesion-api.herokuapp.com/"
        # url_base = "http://localhost:8000/"
        # url_base = "https://api.inveesion.com/"
        url_base = "http://inveesion-api:8000/"
        # url_base = "http://localhost:8000/"

        if guiParam['appType'] == 'Image Application':
            # __, image_byte = DataManager(guiParam).load_image_or_video()
            image_path = DataManager(guiParam).get_image_path()

            if st.button("InVeesion-API"):
                with open(image_path, 'rb') as filelike:
                    files = {"image": ("image", filelike, 'image/jpeg')}
                    endpoint = "image-api/"
                    response = requests.request(
                        'POST', url_base + endpoint, params=guiParam, files=files)

                print(response.url)

                if response.status_code == 200:
                    print('\nRequest is successful: ', response.status_code)

                    st.markdown("## Results")
                    res_json = response.json()['response']
                    keys = list(res_json.keys())
                    values = list(res_json.values())

                    # parse response and extract data (image + csv)
                    with open(paths["received_data"]+'get_demo.png', 'wb') as im_byte:
                        im_byte.write(base64.b64decode(values[0]))
                    with open(paths["received_data"]+'get_demo.csv', 'wb') as csv_byte:
                        csv_byte.write(base64.b64decode(values[1]))

                    st.image(open(paths["received_data"]+'get_demo.png', 'rb').read(),
                             channels="BGR",  use_column_width=True)

                    href = f'<a href="data:file/csv;base64,{values[1]}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                    st.markdown(href, unsafe_allow_html=True)
                    st.dataframe(pd.read_csv(
                        paths["received_data"]+'get_demo.csv'))
                else:
                    print('\nRequest returned an error: ', response.status_code)

        #----------------------------------------------------------------#
        #----------------------------------------------------------------#
        #----------------------------------------------------------------#

        elif guiParam['appType'] == 'Video Application':
            # video_path, video_byte = DataManager(guiParam).load_image_or_video()
            video_path = DataManager(guiParam).get_video_path()

            if st.button("InVeesion-API"):

                with open(video_path, 'rb') as filelike:
                    files = {"video": ("video", filelike, 'video/mp4')}
                    endpoint = "video-api/"
                    response = requests.request(
                        'POST', url_base + endpoint, params=guiParam, files=files)

                print(response.url)

                if response.status_code == 200:
                    print('\nRequest is successful: ', response.status_code)

                    st.markdown("## Results")
                    res_json = response.json()['response']
                    keys = list(res_json.keys())
                    values = list(res_json.values())

                    # parse response and extract data (video + csv)
                    with open(paths["received_data"]+'get_demo.avi', 'wb') as vid_byte:
                        vid_byte.write(base64.b64decode(values[0]))
                    with open(paths["received_data"]+'get_demo.csv', 'wb') as csv_byte:
                        csv_byte.write(base64.b64decode(values[1]))
                    video_path = paths["received_data"]+'get_demo.avi'

                    os.system("ffmpeg -y -i "+video_path+" -vcodec libx264 " +
                              video_path[:-4]+".mp4 && rm "+video_path)

                    with open(video_path[:-4]+'.mp4', 'rb') as f:
                        st.video(f.read())

                    df = pd.read_csv(paths["received_data"]+'get_demo.csv')

                    href = f'<a href="data:file/csv;base64,{values[1]}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                    st.markdown(href, unsafe_allow_html=True)

                    st.dataframe(df)
                    if guiParam['selectedApp']  in ['Object Detection', 'Face Detection']:
                        df_, df_classes = postprocessing_object_detection_df(df)
                        st.dataframe(df_)
                        disp_analytics(df_, df_classes)

                else:
                    print('\nRequest returned an error: ', response.status_code)

    else:
        st.warning("Please select an application")

#----------------------------------------------------------------#
#----------------------------------------------------------------#

if __name__ == "__main__":
    main()
    hide_streamlit_footer()
