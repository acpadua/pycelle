FROM python:3.7.4
RUN pip3 install numpy && pip3 install scikit-image && pip3 install pandas && pip3 install opencv-python && pip3 install pudb
COPY analyse_images.py /
