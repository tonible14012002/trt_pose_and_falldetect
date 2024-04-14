run-demo-fall-vid:
	python main.py --model densenet --video ${dir}

# run-demo-cpu:
# 	@python main.py -C './scripts/samples/fall-vid.mp4' --device cpu

# run-camera-cpu:
# 	@python main.py -C 0 --device cpu

run-demo-cuda:
	@python movenet.py -C './trt_pose/tasks/human_pose/fall-vid.mp4' --device cuda
	
# run-camera-cuda:
# 	@python main.py -C 0 --device cuda