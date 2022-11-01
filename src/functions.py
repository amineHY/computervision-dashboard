import streamlit as st
import youtube_dl
import urllib
import pandas as pd
import os
import plotly.express as px
from collections import Counter
from plotly.subplots import make_subplots
import plotly.graph_objects as go



# import matplotlib.pyplot as plt
# import cv2 as cv
# import numpy as np
# import youtube_dl



# ----------------------------------------------------------------#
# Classes
# ----------------------------------------------------------------#


class GUI:
    """
    This class is dedicated to manage to user interface of the website. It contains methods to edit the sidebar for the selected application as well as the front page.
    """

    def __init__(self):
        """
        Initialization function
        """
        self.list_of_apps = [
            "Empty",
            "Face Mask Detection",
            "Face Detection with Blurring",
            "Face Detection",
            "Object Detection",
            "Fire Detection",
            "Heatmap Motion",
        ]

        self.guiParam = {}

    # ----------------------------------------------------------------

    def getGuiParameters(self):
        """A function that allow to get the Graphical User interface configuration parameters

        Returns:
            guiParam: A dictionary that contains configuration parameter from the sidebar
        """
        self.common_config()
        self.appDescription()
        guiParam = self.guiParam

        return guiParam

    # ----------------------------------------------------------------

    def common_config(self, title="🚀 Computer Vision Dashboard 🚀"):
        """This function returns the configuration parameter from User Interface's Sidebar

        Args:
            title (str, optional): This is the title displayed on the dashboard. Defaults to "Computer Vision Dashboard 🚀".
        """

        # Insert a main title to the Dashboard
        st.title(title)

        # Insert a title of the sidebar
        st.sidebar.title("Configuration")

        # Get the application type from the GUI
        self.appType = st.sidebar.radio(
            "Select an Application", ["Image Application", "Video Application"]
        )

        self.dataSource = st.sidebar.radio(
            "Load data from", ["Database", "URL", "Upload"]
        )

        if self.appType == "Video Application":
            self.recordOutputVideo = st.sidebar.checkbox(
                "Record Video with Overlay", value=True
            )

            self.frameFreq = st.sidebar.slider(
                "Frame Frequency", value=15, min_value=1, max_value=60, step=1
            )

            self.frameMax = st.sidebar.slider(
                "Frames to process",
                value=100,
                min_value=self.frameFreq,
                max_value=500,
                step=1,
            )

        elif self.appType == "Image Application":
            self.recordOutputVideo = False
            self.frameMax = 1
            self.frameFreq = 1

        # Get the selectedApp from the GUI
        self.selectedApp = st.sidebar.selectbox(
            "Apply a Computer Vision Model", self.list_of_apps
        )

        if self.selectedApp == "Empty":
            st.sidebar.warning("Please select an application from the list")

        # Update the guiParam dictionary
        self.guiParam.update(
            dict(
                selectedApp=self.selectedApp,
                appType=self.appType,
                dataSource=self.dataSource,
                recordOutputVideo=self.recordOutputVideo,
                frameMax=self.frameMax,
                frameFreq=self.frameFreq,
            )
        )

        # --------------------------------------------------------------------------

    def appDescription(self):

        st.header("{}".format(self.selectedApp))

        if self.selectedApp == "Object Detection":
            st.info(
                "This application performs object detection using advanced deep learning models. It can detects more than 80 objects (see the full list in COCO dataset)."
            )
            self.sidebarObjectDetection()

        elif self.selectedApp == "Face Detection":
            st.info(
                "This application performs face detection using advanced deep learning models. It can detects faces in images/videos."
            )
            self.sidebarFaceDetection()

        elif self.selectedApp == "Face Detection with Blurring":
            st.info(
                "This application performs face detection using advanced deep learning models. It can detects faces in image/videos. In addition, to preserve privacy, it blurs the detected faces to anonymize."
            )
            self.sidebarFaceDetectionWithBlur()

        elif self.selectedApp == "Fire Detection":
            st.info(
                "This application performs fire detection using advanced deep learning models."
            )
            self.sidebarFireDetection()

        elif self.selectedApp == "Face Mask Detection":
            st.info("This application performs Face Mask Detection")
            self.sidebarFaceMaskDetection()

        elif self.selectedApp == "Heatmap Motion":
            st.info(
                "This application performs heatmap motion. It detect part of the video where there a concentrated movement."
            )
            self.sidebarHeatmapMotion()

        else:
            st.info(
                "To run the computer vision dashboard you must first select an Application from the sidebar menu (other than Empty)"
            )

    # --------------------------------------------------------------------------
    def sidebarEmpty(self):
        pass

    # --------------------------------------------------------------------------

    def sidebarHeatmapMotion(self):

        pass

    # --------------------------------------------------------------------------

    def sidebarFaceDetection(self):
        """
        This function update the dictionary guiParam (from the self class) with parameters of FaceDetection App
        """

        model = st.sidebar.selectbox(
            label="Select available model", options=(["MobileNetSSD"])
        )

        
        confThresh = st.sidebar.slider(
            "Confidence", value=0.60, min_value=0.0, max_value=1.00, step=0.05
        )

        self.guiParam.update(dict(confThresh=confThresh, model=model))

    # --------------------------------------------------------------------------

    def sidebarFaceDetectionWithBlur(self):
        """
        This function update the dictionary guiParam (from the self class) with parameters of FaceDetectionWithBlurring App
        """

        model = st.sidebar.selectbox(
            label="Select available model", options=(["MobileNetSSD"])
        )

        
        confThresh = st.sidebar.slider(
            "Confidence", value=0.60, min_value=0.0, max_value=1.00, step=0.05
        )
        self.guiParam.update(dict(confThresh=confThresh, model=model))

    # --------------------------------------------------------------------------

    def sidebarFaceMaskDetection(self):
        """
        This function update the dictionary guiParam (from the self class) with parameters of FaceMaskDetection App
        """

        model = st.sidebar.selectbox(
            label="Select available model", options=(["MobileNetSSD"])
        )

        
        confThresh = st.sidebar.slider(
            "Confidence", value=0.60, min_value=0.0, max_value=1.00, step=0.05
        )

        self.guiParam.update(dict(confThresh=confThresh, model=model))

    # --------------------------------------------------------------------------

    def sidebarObjectDetection(self):
        """
        This function update the dictionary guiParam (from the self class) with parameters of FaceDetection App
        """
        
        model = st.sidebar.selectbox(
            label="Select available model",
            options=["Caffe-MobileNetSSD", "Darknet-YOLOv3-tiny", "Darknet-YOLOv3"],
        )

        st.sidebar.markdown("### Object Filtering")
        # ------------------------------------------------------#
        allowedLabel = st.sidebar.multiselect(
            label="What object would like to detect?",
            options=("person", "car", "bicycle", "dog", "cell phone", "plane", "fire"),
        )

        allowedLabel = ["all"] if len(allowedLabel) == 0 else allowedLabel

        
        confThresh = st.sidebar.slider(
            "Confidence", value=0.6, min_value=0.0, max_value=1.0
        )
        nmsThresh = st.sidebar.slider(
            "Non-maximum suppression",
            value=0.30,
            min_value=0.0,
            max_value=1.00,
            step=0.05,
        )

        self.guiParam.update(
            dict(
                confThresh=confThresh,
                nmsThresh=nmsThresh,
                model=model,
                allowedLabel=allowedLabel,
            )
        )

    # --------------------------------------------------------------------------

    def sidebarFireDetection(self):
        """
        This function update the dictionary guiParam (from the self class) with parameters of FireDetection App
        """
        model = st.sidebar.selectbox(
            label="Select available model", options=["Darknet-YOLOv3-tiny"]
        )

        
        confThresh = st.sidebar.slider(
            "Confidence", value=0.6, min_value=0.0, max_value=1.0
        )
        nmsThresh = st.sidebar.slider(
            "Non-maximum suppression",
            value=0.30,
            min_value=0.0,
            max_value=1.00,
            step=0.05,
        )

        self.guiParam.update(
            dict(confThresh=confThresh, nmsThresh=nmsThresh, model=model)
        )

    # --------------------------------------------------------------------------

    def sidebarCarsCounting(self):
        """
        This function update the dictionary guiParam (from the self class) with parameters of CarCounting App
        """
        
        model = st.sidebar.selectbox(
            label="Select available model", options=("Model 1", "Model 2", "Model 3")
        )

        self.guiParam.update(dict(model=model))


class DataManager:
    def __init__(self, guiParam):
        self.guiParam = guiParam

        self.url_demo_videos = {
            "Driving car in a city": "https://www.youtube.com/watch?v=7BjNbkONCFw",
            "A Sample Video with Faces": "https://www.youtube.com/watch?v=ohmajJTcpNk",
        }

        self.url_demo_images = {
            "image NY-City": "https://s4.thingpic.com/images/8a/Qcc4eLESvtjiGswmQRQ8ynCM.jpeg",
            "image Paris-street": "https://www.discoverwalks.com/blog/wp-content/uploads/2018/08/best-streets-in-paris.jpg",
            "image people": "https://www.rembrandtmall.co.za/wp-content/uploads/2019/05/people-1.jpg"
        }

        self.demo_video_examples = {
            "Street-CCTV": guiParam["path_database"] + "object.mp4",
            "Showroom": guiParam["path_database"] + "showroom.mov",
        }
        self.demo_image_examples = {
            "image COVID-19 Mask": guiParam["path_database"] + "face_mask.jpeg",
            "image Family-picture": guiParam["path_database"] + "family.jpg",
            "image Dog": guiParam["path_database"] + "dog.jpg",
            "image Crosswalk": guiParam["path_database"] + "demo.jpg",
            "image Car on fire": guiParam["path_database"] + "car_on_fire.jpg",
        }

        self.image = None
        self.image_byte = None

        self.video = None
        self.video_byte = None
        self.data = None
        self.data_byte = None

    # --------------------------------------------------------#

    def get_video_path(self):

        if self.guiParam["dataSource"] == "Upload":
            filelike = st.file_uploader(
                "Upload a video (200 Mo maximum) ...", type=["mp4", "mpeg", "avi"]
            )

            video_path = "database/uploaded_video.png"

            if filelike:
                with open(video_path, "wb") as f:
                    f.write(filelike.read())
                return video_path
            else:
                return None

            # ------------------------------------------------------#

        elif self.guiParam["dataSource"] == "Database":

            video_path = st.text_input("Enter PATH of the video")

            if os.path.isfile(video_path):
                return video_path

            else:
                video_path_idx = st.selectbox(
                    "Or select a demo video from the list",
                    list(self.demo_video_examples.keys()),
                )
                video_path = self.demo_video_examples[video_path_idx]
                return video_path

            # ------------------------------------------------------#

        elif self.guiParam["dataSource"] == "URL":

            video_url = st.text_input("Enter URL of the video")
            video_path = "database/downloaded_video.mp4"

            if video_url != "":
                isinstance(video_url, str)
                print("Downloading ", video_url)
                ydl_opts = dict(format="bestvideo[height<=480]", outtmpl=video_path)
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                return video_path

            else:
                st.info(
                    "Here are some video samples"
                    + "\n Driving car in a city: https://www.youtube.com/watch?v=7BjNbkONCFw \
                \n A Sample Video with Faces: https://www.youtube.com/watch?v=ohmajJTcpNk"
                )
                video_url = "https://www.youtube.com/watch?v=7BjNbkONCFw"

            # ------------------------------------------------------#

        else:
            raise ValueError("Please select a 'Data Source' from the list")

    def get_image_path(self):
        # --------------------------------------------#
        if self.guiParam["dataSource"] == "Database":
            image_path = st.text_input("Enter the image PATH (for local deployment)")

            if image_path == "":
                image_path_idx = st.selectbox(
                    "Or select a demo image from the list",
                    list(self.demo_image_examples.keys()),
                )
                image_path = self.demo_image_examples[image_path_idx]
            return image_path

        # --------------------------------------------#

        elif self.guiParam["dataSource"] == "Upload":
            filelike = st.file_uploader("Upload an image", type=["png", "jpg"])
            image_path = "database/uploaded_image.png"

            if filelike != None:
                with open(image_path, "wb") as f:
                    f.write(filelike.read())
                return image_path
            else:
                raise ValueError("Please Upload an image first")

        # --------------------------------------------#

        elif self.guiParam["dataSource"] == "URL":
            url_image = st.text_input("Enter the image URL")
            image_path = "database/downloaded_image.png"

            if url_image != "":
                with open(image_path, "wb") as f:
                    f.write(urllib.request.urlopen(url_image).read())
                    return image_path
            else:
                url_image_idx = st.selectbox(
                    "Or select a URL from the list", list(self.url_demo_images.keys())
                )
                url_image = self.url_demo_images[url_image_idx]
                with open(image_path, "wb") as f:
                    f.write(urllib.request.urlopen(url_image).read())
                    return image_path



    # def load_image_source(self):
    #     """ """

    #     if self.guiParam["dataSource"] == "Database":

    #         @st.cache(allow_output_mutation=True)
    #         def load_image_from_path(image_path):
    #             # im_rgb = cv.imread(image_path, cv.IMREAD_COLOR)
    #             with open(image_path, "rb") as f:
    #                 im_byte = f.read()
    #                 im_ndarr = np.frombuffer(im_byte, dtype=np.uint8)
    #                 im_rgb = cv.imdecode(im_ndarr, cv.IMREAD_COLOR)
    #                 return im_rgb, f

    #             image_path

    #         file_path = st.text_input("Enter the image PATH")

    #         if os.path.isfile(file_path):
    #             self.image, self.image_byte = load_image_from_path(image_path=file_path)

    #         elif file_path == "":
    #             file_path_idx = st.selectbox(
    #                 "Or select a demo image from the list",
    #                 list(self.demo_image_examples.keys()),
    #             )
    #             file_path = self.demo_image_examples[file_path_idx]

    #             self.image, self.image_byte = load_image_from_path(image_path=file_path)
    #         else:
    #             raise ValueError("[Error] Please enter a valid image path")

    #         # --------------------------------------------#
    #         # --------------------------------------------#

    #     elif self.guiParam["dataSource"] == "Upload":

    #         @st.cache(allow_output_mutation=True)
    #         def load_image_from_upload(file):
    #             filelike = file
    #             # im_ndarr = np.frombuffer(im_byte, dtype=np.uint8)
    #             # im_rgb = cv.imdecode(im_ndarr, cv.IMREAD_COLOR)
    #             im_rgb = []
    #             return im_rgb, filelike

    #         file_path = st.file_uploader("Upload an image", type=["png", "jpg"])

    #         if file_path != None:
    #             self.image, self.image_byte = load_image_from_upload(file_path)
    #         elif file_path == None:
    #             raise ValueError("[Error] Please upload a valid image ('png', 'jpg')")
    #         # --------------------------------------------#
    #         # --------------------------------------------#

    #     elif self.guiParam["dataSource"] == "URL":

    #         @st.cache(allow_output_mutation=True)
    #         def load_image_from_url(url_image):
    #             print("Downloading ...")
    #             file = urllib.request.urlopen(url_image)
    #             im_byte = file.read()
    #             # tmp = np.asarray(bytearray(file.read()), dtype=np.uint8)
    #             im_ndarr = np.frombuffer(im_byte, dtype=np.uint8)
    #             im_rgb = cv.imdecode(im_ndarr, cv.IMREAD_COLOR)
    #             return im_rgb, im_byte

    #         file_path = st.text_input("Enter the image URL")

    #         if file_path != "":
    #             self.image, self.image_byte = load_image_from_url(url_image=file_path)

    #         elif file_path == "":

    #             file_path_idx = st.selectbox(
    #                 "Or select a URL from the list", list(self.url_demo_images.keys())
    #             )
    #             file_path = self.url_demo_images[file_path_idx]

    #             self.image, self.image_byte = load_image_from_url(url_image=file_path)
    #         else:
    #             raise ValueError("[Error] Please enter a valid image URL")

    #         # --------------------------------------------#
    #         # --------------------------------------------#

    #     else:
    #         raise ValueError("Please select one source from the list")

    #     return self.image, self.image_byte

    # def load_image_or_video(self):
    #     """
    #     Handle the data input from the user parameters
    #     """
    #     if self.guiParam["appType"] == "Image Application":
    #         self.data, self.data_byte = self.load_image_source()

    #     elif self.guiParam["appType"] == "Video Application":
    #         self.data, self.data_byte = self.load_video_source()

    #     else:
    #         raise ValueError("[Error] Please select of the two Application pipelines")

    #     return self.data, self.data_byte

    # def check_video_object(self, video):
    #     """ """
    #     if video != None and video.isOpened():
    #         print("[INFO] File is correctly openned")

    #     else:
    #         raise Exception(
    #             "[Error] Could not open the video source: {}".format(
    #                 self.guiParam["dataSource"]
    #             )
    #         )

    # def load_video_source(self):
    #     """ """
    #     # print(self.guiParam["dataSource"])
    #     if self.guiParam["dataSource"] == "None":
    #         self.video = None
    #         st.warning("No application is selected.")

    #         # ------------------------------------------------------#
    #         # ------------------------------------------------------#
    #     elif self.guiParam["dataSource"] == "Upload":

    #         @st.cache(allow_output_mutation=True)
    #         def load_video_from_upload(file_path):
    #             return None, file_path

    #         file_path = st.file_uploader(
    #             "Upload a video (200 Mo maximum) ...", type=["mp4", "mpeg", "avi"]
    #         )

    #         if file_path != "":
    #             self.video, self.video_byte = load_video_from_upload(file_path)
    #         else:
    #             st.info("Please upload a valid image ('mp4', 'mpeg', 'avi')")

    #         # if video_url != "":
    #         #     self.video, self.video_byte = load_video_from_url(video_url)
    #         # else:
    #         #     st.info("Here are some video samples"+
    #         #     "\n Driving car in a city: https://www.youtube.com/watch?v=7BjNbkONCFw \
    #         #     \n A Sample Video with Faces: https://www.youtube.com/watch?v=ohmajJTcpNk"
    #         #     )
    #         # ------------------------------------------------------#
    #         # ------------------------------------------------------#

    #     elif self.guiParam["dataSource"] == "Database":

    #         @st.cache(allow_output_mutation=True)
    #         def load_video_from_path(video_path):
    #             isinstance(video_path, str)
    #             video = cv.VideoCapture(video_path)
    #             with open(video_path, "rb") as f:
    #                 video_byte = f.read()
    #                 # print(type(video_byte))
    #             return video_path, video_byte

    #         file_path = st.text_input("Enter PATH of the video")

    #         if os.path.isfile(file_path):
    #             self.video, self.video_byte = load_video_from_path(file_path)
    #         elif file_path == "":
    #             file_path_idx = st.selectbox(
    #                 "Or select a demo image from the list",
    #                 list(self.demo_video_examples.keys()),
    #             )
    #             file_path = self.demo_video_examples[file_path_idx]
    #             self.video, self.video_byte = load_video_from_path(video_path=file_path)

    #         else:
    #             raise ValueError("[Error] Please enter a valid video path")

    #         # ------------------------------------------------------#
    #         # ------------------------------------------------------#

    #     # elif self.guiParam["dataSource"] == 'Webcam':

    #     #     @st.cache(allow_output_mutation=True)
    #     #     def load_video_from_webcam(webcam_id):
    #     #         isinstance(webcam_id, int)
    #     #         video = cv.VideoCapture(webcam_id)
    #     #         from time import time
    #     #         time.sleep(2)
    #     #         return video

    #     #     webcam_id = 0  # default value, change it to the suitable device
    #     #     self.video = load_video_from_webcam(webcam_id)

    #     # ------------------------------------------------------#
    #     # ------------------------------------------------------#

    #     elif self.guiParam["dataSource"] == "URL":

    #         @st.cache(allow_output_mutation=True)
    #         def load_video_from_url(video_url):
    #             isinstance(video_url, str)
    #             print("Downloading ", video_url)

    #             ydl_opts = {
    #                 "format": "bestvideo[height<=480]",
    #                 "outtmpl": "database/tmp.mp4",
    #             }
    #             with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    #                 ydl.download([video_url])
    #             print(ydl)
    #             # video = pafy.new(video_url)
    #             # videoHightRes = video.getbest(preftype="mp4")
    #             # videoHightRes.download('database/demso.mp4')
    #             with open("database/tmp.mp4", "rb") as f:
    #                 video_byte = f.read()
    #                 print("reading video")
    #             os.system("rm database/tmp.mp4")

    #             video = None
    #             return video, video_byte

    #         video_url = st.text_input("Enter URL of the video")
    #         # st.info(
    #         #     'Samples here: https://research.google.com/youtube8m/explore.html')

    #         if video_url != "":
    #             self.video, self.video_byte = load_video_from_url(video_url)
    #         else:
    #             st.info(
    #                 "Here are some video samples"
    #                 + "\n Driving car in a city: https://www.youtube.com/watch?v=7BjNbkONCFw \
    #             \n A Sample Video with Faces: https://www.youtube.com/watch?v=ohmajJTcpNk"
    #             )
    #         # elif video_url == "":
    #         #     video_url_idx = st.selectbox(
    #         #         'Or select a URL from the list', list(self.url_demo_videos.keys()))
    #         #     video_url = self.url_demo_videos[video_url_idx]
    #         #     self.video, self.video_byte = load_video_from_url(video_url)

    #         # else:
    #         #     raise ValueError("[Error] Please enter a valid video URL")

    #         # ------------------------------------------------------#

    #     else:
    #         raise ValueError("Please select a 'Data Source' from the list")

    #     # self.check_video_object(self.video)

    #     return self.video, self.video_byte


# ----------------------------------------------------------------#
# Python functions
# ----------------------------------------------------------------#


def postprocessing_object_detection_df(df):
    """_summary_

    Args:
        df (DataFrame): _description_

    Returns:
        _type_: _description_
    """

    df_ = df.copy()

    # Unwrap bboxes
    df_.bboxes = df.bboxes.apply(pd.eval)
    df_.confidences = df.confidences.apply(pd.eval)
    df_.predClasses = df.predClasses.apply(pd.eval)

    if "predClasses" in df_.columns:
        df_.loc[:, "counting_obj"] = (
            df_["predClasses"].apply(Counter).values
        )
        df_.loc[:, "objectClass"] = (
            df_.loc[:, "counting_obj"].apply(lambda x: list(x.keys())).values
        )
        df_.loc[:, "objectNumb"] = (
            df_.loc[:, "counting_obj"].apply(lambda x: list(x.values())).values
        )

        df_classes = pd.DataFrame(df_.counting_obj.to_dict()).T
        dataf = df_.join(df_classes)

    return dataf, df_classes


def disp_analytics(df, df_classes):
    """This function allow display analytics that were gathered after applying a deep learning model

    Args:
        df (DataFrame): This is a pandas dataframe that contains extracted analytics
        df_classes (DataFrame): _description_
    """
    if len(df_classes.columns) > 0:
        st.markdown("## Global Analytics")

        # Add a bar chart
        fig = px.bar(x=df_classes.columns, y=df_classes.sum())
        fig.update_layout(
            height=400,
            width=900,
            title_text="...",
            yaxis=dict(title_text="Number of Detection"),
            xaxis=dict(title_text="Detection Object in the Video"),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)

        # Add Pie chart
        fig = px.pie(
            df_classes,
            values=df_classes.sum(),
            names=df_classes.columns,
            title="Detected Objects in the Video",
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)

        # Add a subplot of scatter plots
        fig = make_subplots(rows=len(df_classes.columns), cols=1)
        for idx, feat in enumerate(df_classes.columns):
            fig.add_trace(
                go.Scatter(x=df.frameIdx, y=df[feat], mode="lines+markers", name=feat),
                row=idx + 1,
                col=1,
            )
        tmp = (len(df_classes.columns)) * 400
        fig.update_layout(height=tmp, width=900, title_text="Objects Filtering")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        st.plotly_chart(fig)

        fig = px.scatter(x=df.frameIdx, y=df.total_object)
        fig.update_layout(height=400, width=900, title_text="Total Detection per frame")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)

        st.markdown("## Motion Analytics")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(
            go.Scatter(x=df.frameIdx, y=df.motion_status, mode="lines"), row=1, col=1
        )
        fig.update_layout(
            height=600,
            width=900,
            title_text="Detected Motion in the Video",
            yaxis=dict(title_text="Motion Status"),
            xaxis=dict(title_text="Timestamp"),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)
