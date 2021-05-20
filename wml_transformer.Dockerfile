FROM python:3.7-slim

RUN apt-get update \
&& apt-get install -y --no-install-recommends git

COPY . .
RUN pip install pip==20.2
RUN pip install flask
RUN pip install tensorflow
RUN pip install -e .

EXPOSE 5000
CMD ["python", "/wml_transformer/main.py"]
