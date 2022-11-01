# ----------------------------------------------------------------#
"""
Author: Amine Hadj-Youcef
Email: hadjyoucef.amine@gmail.com
Github: https://github.com/amineHY/computervision-dashboard.git
"""
# ----------------------------------------------------------------#
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
import src.functions as F

print(__doc__)

# ----------------------------------------------------------------#
# python functions
# ----------------------------------------------------------------#


def main():
    """
    This is the main function
    """
    # ----------------------------------------------------------------#
    # Get parameters from the GUI
    # ----------------------------------------------------------------#
    guiParam = F.GUI().getGuiParameters()

    # Define paths
    paths = {
        "path_database": "database/",
        "received_data": "received_data/",
    }

    # Update the dictionary of parameters
    guiParam.update(paths)

    # ----------------------------------------------------------------#
    # Send Request to fastAPI
    # ----------------------------------------------------------------#

    # Check if the selectedApp is not empty
    if guiParam["selectedApp"] != "Empty":

        # ----------------------------------------------------------------#
        # Set the API URL
        # ----------------------------------------------------------------#
        api_url_base = (
            "http://localhost:8000/"  # api_url_base = "http://inveesion-api:8000/"
        )

        # If the selectedApp is for images
        # ----------------------------------------------------------------#
        if guiParam["appType"] == "Image Application":
            # Get the image_path depending on data source
            image_path = F.DataManager(guiParam).get_image_path()

            # Trigger the API only if the button 'RUN' is pressed
            if st.button("Run"):

                with open(image_path, "rb") as filelike:
                    print("Sending request to FastAPI...")
                    files = {"image": ("image", filelike, "image/jpeg")}
                    endpoint = "image-api/"
                    response = requests.request(
                        "POST", api_url_base + endpoint, params=guiParam, files=files
                    )
                print("[FastAPI Response] : ", response.url)

                # Process the response from the API
                if response.status_code == 200:
                    print("\n[API] Success: ", response.status_code)

                    st.markdown("## Results")
                    response_json = response.json()["response"]
                    values = list(response_json.values())

                    # Parse the API response and extract data (image + csv)
                    img_overlay_path = paths["received_data"] + "img_overlay"
                    with open(img_overlay_path, "wb") as im_byte:
                        im_byte.write(base64.b64decode(values[0]))
                    with open(
                        paths["received_data"] + "csv_analytics.csv", "wb"
                    ) as csv_byte:
                        csv_byte.write(base64.b64decode(values[1]))

                    st.image(
                        open(img_overlay_path, "rb").read(),
                        channels="BGR",
                        use_column_width=True,
                    )

                    href = f'<a href="data:file/csv;base64,{values[1]}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                    st.markdown(href, unsafe_allow_html=True)
                    st.dataframe(
                        pd.read_csv(paths["received_data"] + "csv_analytics.csv")
                    )
                else:
                    print("\[API] Failure: ", response.status_code)

        # If the selectApp is for videos
        # ----------------------------------------------------------------#

        elif guiParam["appType"] == "Video Application":
            # Get the video_path depending on data source
            video_path = F.DataManager(guiParam).get_video_path()

            # Trigger the API only if the button 'RUN' is pressed
            if st.button("Run"):

                with open(video_path, "rb") as filelike:
                    print("Sending request to FastAPI...")
                    files = {"video": ("video", filelike, "video/mp4")}
                    endpoint = "video-api/"
                    response = requests.request(
                        "POST", api_url_base + endpoint, params=guiParam, files=files
                    )
                print("[FastAPI Response] : ", response.url)

                # Process the response from the API
                if response.status_code == 200:
                    print("\n[API] Success: ", response.status_code)

                    st.markdown("## Results")
                    response_json = response.json()["response"]
                    values = list(response_json.values())

                    # Parse the API response and extract data (video + csv)
                    vid_overlay_path = paths["received_data"] + "vid_overlay.avi"
                    with open(vid_overlay_path, "wb") as vid_byte:
                        vid_byte.write(base64.b64decode(values[0]))
                    csv_path = paths["received_data"] + "csv_analytics.csv"
                    with open(csv_path, "wb") as csv_byte:
                        csv_byte.write(base64.b64decode(values[1]))

                    # Convert video to mp4
                    # ----------------------------------------------------------------#
                    os.system(
                        "ffmpeg -y -i "
                        + vid_overlay_path
                        + " -vcodec libx264 "
                        + vid_overlay_path[:-4]
                        + ".mp4 && rm "
                        + vid_overlay_path
                    )

                    with open(vid_overlay_path[:-4] + ".mp4", "rb") as f:
                        st.video(f.read())

                    href = f'<a href="data:file/csv;base64,{values[1]}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                    st.markdown(href, unsafe_allow_html=True)

                    # Display analytics
                    # ----------------------------------------------------------------#
                    if guiParam["selectedApp"] in [
                        "Object Detection",
                        "Face Detection",
                    ]:
                        df = pd.read_csv(csv_path)
                        df_, df_classes = F.postprocessing_object_detection_df(df)
                        st.dataframe(df_)
                        F.disp_analytics(df_, df_classes)

                else:
                    print("\n[API] Failure: ", response.status_code)

    else:
        st.warning("Please select an application from the sidebar menu")


# ----------------------------------------------------------------#
# main program
# ----------------------------------------------------------------#

if __name__ == "__main__":
    main()
