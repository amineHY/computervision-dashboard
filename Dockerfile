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
    pip install --no-cache-dir -r requirements.txt &&\
	rm -rf requirements.txt

EXPOSE 8080

# --------------- Export envirennement variable ---------------
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["streamlit", "run", "--server.port", "8080", "main.py"]    
