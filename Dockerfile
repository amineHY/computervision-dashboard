FROM python:3.10.8

LABEL maintainer "Amine Hadj-Youcef  <hadjyoucef.amine@gmail.com>"
# www.amine-hy.com

# --------------- Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME

# --------------- Install packages ---------------
RUN apt-get update -qq && apt-get -y install \
    ffmpeg

# --------------- Install python packages using `pip` ---------------
COPY requirements.txt $APP_HOME
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --upgrade 

# --------------- Export envirennement variable ---------------
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY . ./

# --------------- Streamlit-specific commands


# RUN mkdir -p /root/.streamlit
# RUN bash -c 'echo -e "\
# [general]\n\
# email = \"\"\n\
# " > /root/.streamlit/credentials.toml'

# RUN bash -c 'echo -e "\
# [server]\n\
# enableCORS = false\n\
# " > /root/.streamlit/config.toml'

# EXPOSE 8501

# --------------- Run the image as a non-root user: Heroku
# RUN useradd -m myuser
# USER myuser

# --------------- Run the app
CMD streamlit run main.py --server.port $PORT --server.enableCORS false
