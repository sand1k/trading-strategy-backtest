FROM continuumio/miniconda3:latest

WORKDIR /backtest

COPY environment.yml .
RUN conda env create -f environment.yml

COPY ./sharadar_ingest.py ./backtest_linear.py /backtest
COPY ./extension.py /root/.zipline/
RUN ln -s /backtest/sharadar_ingest.py /root/.zipline/

#CMD [ \
#  "conda", "run", "--no-capture-output", "-n", "zipline", \
#  "python sharadar_ingest.py" \
#]

CMD ["/bin/bash", "-c", "conda run --no-capture-output -n zipline zipline ingest -b sharadar && conda run --no-capture-output -n zipline python backtest_linear.py"]