import streamlit as st
import youtube_dl
import urllib
import pandas as pd
import os
import plotly.express as px
from collections import Counter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import logging


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
            "image people": "https://www.rembrandtmall.co.za/wp-content/uploads/2019/05/people-1.jpg",
        }

        self.demo_video_examples = {
            "video Street-CCTV": guiParam["path_database"] + "object.mp4",
            "video Showroom": guiParam["path_database"] + "showroom.mov",
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
                logging.info("Downloading ", video_url)
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

# ----------------------------------------------------------------#
# Python functions
# ----------------------------------------------------------------#


def postprocessing_object_detection_df(df_silver):
    """_summary_

    Args:
        df_silver (DataFrame): _description_

    Returns:
        _type_: _description_
    """

    df_gold = df_silver.copy()

    # Unwrap bounding boxes (bboxes)
    df_gold["bboxes"] = df_gold["bboxes"].apply(pd.eval)
    df_gold["confidences"] = df_gold["confidences"].apply(pd.eval)
    df_gold["predClasses"] = df_gold["predClasses"].apply(pd.eval)

    if "predClasses" in df_gold.columns:
        df_gold.loc[:, "counting_obj"] = df_gold["predClasses"].apply(Counter).values
        df_gold.loc[:, "object_class"] = (
            df_gold.loc[:, "counting_obj"].apply(lambda x: list(x.keys())).values
        )
        df_gold.loc[:, "object_count"] = (
            df_gold.loc[:, "counting_obj"].apply(lambda x: list(x.values())).values
        )
        df_gold = df_gold.join(pd.DataFrame(df_gold["counting_obj"].to_dict()).T)
    return df_gold


def plot_analytics(df_gold):
    """This function allow display analytics that were gathered after applying a deep learning model

    Args:
        df_gold (DataFrame): This is a pandas dataframe that contains extracted analytics
        df_classes (DataFrame): _description_
    """

    df_classes = pd.DataFrame(df_gold["counting_obj"].to_dict()).T

    if len(df_classes.columns) > 0:

        st.markdown("## Global Analytics")
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "bar"}, {"type": "pie"}]],
        )

        # Add a bar chart
        fig.add_trace(go.Bar(x=df_classes.columns, y=df_classes.sum()), row=1, col=1)
        
        # Add Pie chart
        fig.add_trace(
            go.Pie(
                values=df_classes.sum(),
                labels=df_classes.columns,
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            height=400,
            width=900,
            title_text="Detected Objects from the Video",
            yaxis=dict(title_text="# of Detection"),
            xaxis=dict(title_text="Detected Objects"),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        st.plotly_chart(fig)



        # Plot total detection per frame
        fig = px.scatter(x=df_gold["frameIdx"], y=df_gold["total_object"])
        fig.update_layout(height=500, width=900, title_text="Total Detection per frame")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)

        # Add a subplot per type of object
        fig = make_subplots(
            rows=len(df_classes.columns),
            cols=1,
            shared_xaxes=True,
            subplot_titles=list(df_classes.columns),
        )
        for idx, feat in enumerate(df_classes.columns):
            fig.add_trace(
                go.Scatter(
                    x=df_gold["frameIdx"],
                    y=df_gold[feat],
                    mode="markers",
                    name=feat,
                ),
                row=idx + 1,
                col=1,
            )
        tmp = (len(df_classes.columns)) * 300
        fig.update_layout(height=tmp, width=900, title_text="Objects Filtering")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        st.plotly_chart(fig)

        st.markdown("## Motion Analysis")
        # ----------------------------------------------------------------#

        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(
            go.Scatter(x=df_gold["processed_on"], y=df_gold["motion_status"], mode="lines"),
            row=1,
            col=1,
        )

        fig.update_layout(
            height=500,
            width=900,
            title_text="Detected Motion in the Video",
            yaxis=dict(title_text="Motion Status"),
            xaxis=dict(title_text="Timestamp"),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)
