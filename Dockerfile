FROM python:3.7


LABEL maintainer "Amine Hadj-Youcef  <hadjyoucef.amine@gmail.com>"
# www.amine-hy.com

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# --------------- # Install packages ---------------
# RUN apt-get update -qq && apt-get -y install \
#     ffmpeg

# --------------- Install python packages using `pip` ---------------

RUN pip install --upgrade pip && \
    pip install -r requirements.txt 
    #--no-cache-dir 
	# rm -rf requirements.txt

# EXPOSE 8080

# --------------- Export envirennement variable ---------------
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# ENV PORT=PORT


# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
port = $PORT\n\
" > /root/.streamlit/config.toml'
 
# EXPOSE 8501

# Run the image as a non-root user
RUN useradd -m myuser
USER myuser

# Run the app. 
CMD streamlit run main.py --server.port $PORT --server.enableCORS false