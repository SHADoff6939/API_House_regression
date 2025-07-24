FROM ubuntu:latest
LABEL authors="rkush"

ENTRYPOINT ["top", "-b"]