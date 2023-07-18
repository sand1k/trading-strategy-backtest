FROM continuumio/miniconda3:latest

WORKDIR /backtest

COPY environment.yml .
RUN conda env create -f environment.yml

COPY ./src/ ./run.sh /backtest
COPY ./extension.py /root/.zipline/
RUN ln -s /backtest/sharadar_ingest.py /root/.zipline/

RUN chmod +x /backtest/run.sh

CMD ["/bin/bash", "-c", "conda run --no-capture-output -n zipline ./run.sh"]