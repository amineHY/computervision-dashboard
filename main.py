# ----------------------------------------------------------------#
"""
Author: Amine Hadj-Youcef
Email: hadjyoucef.amine@gmail.com
GitHub: https://github.com/amineHY/computervision-dashboard.git
"""
# ----------------------------------------------------------------#


import base64
import os

import pandas as pd
import requests
import streamlit as st
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
                    api_endpoint = "image-api/"
                    response = requests.request(
                        "POST",
                        api_url_base + api_endpoint,
                        params=guiParam,
                        files=files,
                        timeout=360,
                    )
                    print("[FastAPI Response] : ", response.url)

                # Process the response from the API
                if response.status_code == 200:
                    print("\n[API] Success: ", response.status_code)

                    st.markdown("## Results")
                    response_json = response.json()["response"]
                    values = list(response_json.values())

                    # Parse API response and extract data (image + csv)
                    img_overlay_path = paths["received_data"] + "img_overlay"
                    with open(img_overlay_path, "wb") as im_byte:
                        im_byte.write(base64.b64decode(values[0]))
                    
                    csv_path = paths["received_data"] + "csv_analytics.csv"
                    with open(csv_path, "wb") as csv_byte:
                        csv_byte.write(base64.b64decode(values[1]))

                    # Display image with overlay
                    st.image(
                        open(img_overlay_path, "rb").read(),
                        channels="BGR",
                        use_column_width=True,
                    )
                    # Display link to download CSV file
                    href = f'<a href="data:file/csv;base64,{values[1]}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Display dataframe
                    df_silver = pd.read_csv(csv_path)
                    
                    st.markdown('## Display Received Analytics from the API')
                    st.dataframe(df_silver)

                else:
                    print("\n[API] Failure: ", response.status_code)

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
                    api_endpoint = "video-api/"
                    response = requests.request(
                        "POST",
                        api_url_base + api_endpoint,
                        params=guiParam,
                        files=files,
                        timeout=360,
                    )
                    print("[FastAPI Response] : ", response.url)

                # Process the response from the API
                if response.status_code == 200:
                    print("\n[API] Success: ", response.status_code)

                    st.markdown("## Results")
                    response_json = response.json()["response"]
                    values = list(response_json.values())

                    # Parse API response and extract data (video + csv)
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
                    # Display video with overlay
                    with open(vid_overlay_path[:-4] + ".mp4", "rb") as f:
                        st.video(f.read())
                    
                    # Display link to download CSV file
                    href = f'<a href="data:file/csv;base64,{values[1]}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                    st.markdown(href, unsafe_allow_html=True)

                    # Display analytics
                    # ----------------------------------------------------------------#
                    if guiParam["selectedApp"] in [
                        "Object Detection",
                        "Face Detection",
                    ]:
                        df_silver = pd.read_csv(csv_path)
                        df_gold = F.postprocessing_object_detection_df(df_silver)
                        
                        st.markdown('## Display Received Analytics from the API')
                        st.dataframe(df_gold)
                        F.plot_analytics(df_gold)

                else:
                    print("\n[API] Failure: ", response.status_code)

    else:
        st.warning("Please select an application from the sidebar menu")


# ----------------------------------------------------------------#
# main program
# ----------------------------------------------------------------#

if __name__ == "__main__":
    main()
