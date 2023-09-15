install:
	cd thirdparty/mmdetection && python -m pip install -e .
	cd ../.. && python -m pip install -e .
clean:
	rm -r ssod.egg-info
	rm -r thirdparty/mmdetection/mmdet.egg-info
