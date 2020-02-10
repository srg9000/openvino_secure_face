from face_recognition_demo_modified import Visualizer, FrameProcessor
from threading import Thread
import sys
import cv2
import queue

class arguments():

	def __init__(self):
		self.allow_grow = False
		self.cpu_lib = 'F:\\Program Files (x86)\\IntelSWTools\\openvino_2019.3.379\\deployment_tools\\inference_engine\\bin\\intel64\\Release\\cpu_extension_avx2.dll'
		self.crop_height = 0
		self.crop_width = 0
		self.d_fd = 'CPU'
		self.d_lm = 'CPU'
		self.d_reid = 'CPU'
		self.exp_r_fd = 1.15
		self.fg = 'database'
		self.gpu_lib = ''
		self.input = '0'
		self.m_fd = 'F:/Semester 8/GeekyBee/Face_detection_improve/intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml'
		self.m_lm = 'F:/Semester 8/GeekyBee/Face_detection_improve/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml'
		self.m_reid = 'F:/Semester 8/GeekyBee/Face_detection_improve/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml'
		self.no_show = False
		self.output = ''
		self.output_file = 'out.csv'
		self.perf_stats = False
		self.run_detector = False
		self.t_fd = 0.6
		self.t_id = 0.3
		self.timelapse = True
		self.verbose = False
		self.skip_frames = 1
		self.maxsize = 32
		self.signal_queue = queue.Queue()
'''
"F:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\bin\setupvars.bat"
cd F:\Semester 8\GeekyBee\attendence_face_recog
cd
F:
python main_gui_2.py
'''
class inference_object():
	def __init__(self, args):
		# self.args = arguments()
		self.args = args
		self.args.allow_grow = args.allow_grow if args.allow_grow!=None else self.args.allow_grow
		self.args.fg = args.fg
		self.visualizer  = Visualizer(self.args)
		self.stop = False
		print("new object created")

	def start_visualizer(self):
		self.visualizer.run(self.args)


	def get_frame(self):
		while True:
			if self.visualizer.output_queue.qsize()>0:
				frame = self.visualizer.output_queue.get() 
			else:
				continue
		# sys.exit(0)
			if __name__ != '__main__':
				yield frame

def start_visualizer_wrapper(viz_object):
	viz_object.start_visualizer()

def get_frame_wrapper(viz_object):
	viz_object.get_frame()

if __name__ == '__main__':
	viz_object = inference_object()
	visualize_thread = Thread(target = viz_object.start_visualizer).start()
	# show_thread = Thread(target = viz_object.get_frame).start()
	while True:
		cv2.imshow('frame',viz_object.visualizer.output_queue.get() )
		key = cv2.waitKey(0) & 0xFF
		# if key in {ord('q'), ord('Q'), 27}:
		# 	cv2.destroyAllWindows()
		# 	sys.exit(0)

	# visualize_thread.join()
	# show_thread.join()
	#, args = (viz_object,)