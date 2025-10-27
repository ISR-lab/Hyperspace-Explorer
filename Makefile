IMAGE_NAME=balgrist
CONTAINER_NAME=balgrist-container

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -p 8888:8888 -v "$(PWD)":/app $(IMAGE_NAME)

shell:
	docker run -it --rm -v "$(PWD)":/app $(IMAGE_NAME) bash
