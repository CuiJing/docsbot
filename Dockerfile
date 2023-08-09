FROM python:3
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r /app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


# copy code
COPY example_docs /app/example_docs
COPY docsbot /app/docsbot

ENTRYPOINT ["python", "/app/docsbot/cli.py"]
CMD ["listbase"]
