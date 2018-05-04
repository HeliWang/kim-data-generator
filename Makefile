tag=1.0

build-docker:
	docker build -t dstlr/dstlr-ml:${tag} .
push-docker:
	docker push dstlr/dstlr-ml:${tag}
