import cv2
import numpy as np
import imutils
import face_recognition_demo
from argparse import ArgumentParser

DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL']
def build_argparser():
	parser = ArgumentParser()

	general = parser.add_argument_group('General')
	general.add_argument('-i', '--input', metavar="PATH", default=0,
						 help="(optional) Path to the input video "
						 "('0' for the camera, default)")
	general.add_argument('-o', '--output', metavar="PATH", default="",
						 help="(optional) Path to save the output video to")
	general.add_argument('--no_show', action='store_true',
											 help="(optional) Do not display output")
	general.add_argument('-tl', '--timelapse', action='store_true',
						 help="(optional) Auto-pause after each frame")
	general.add_argument('-cw', '--crop_width', default=0, type=int,
						 help="(optional) Crop the input stream to this width "
						 "(default: no crop). Both -cw and -ch parameters "
						 "should be specified to use crop.")
	general.add_argument('-ch', '--crop_height', default=0, type=int,
						 help="(optional) Crop the input stream to this height "
						 "(default: no crop). Both -cw and -ch parameters "
						 "should be specified to use crop.")

	gallery = parser.add_argument_group('Faces database')
	gallery.add_argument('-fg', metavar="PATH", default='database',
						 help="Path to the face images directory")
	gallery.add_argument('--run_detector', action='store_true',
						 help="(optional) Use Face Detection model to find faces"
						 " on the face images, otherwise use full images.")

	models = parser.add_argument_group('Models')
	models.add_argument('-m_fd', metavar="PATH", default="F:/Semester 8/GeekyBee/Face_detection_improve/intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml",
						help="Path to the Face Detection model XML file")
	models.add_argument('-m_lm', metavar="PATH", default="F:/Semester 8/GeekyBee/Face_detection_improve/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml",
						help="Path to the Facial Landmarks Regression model XML file")
	models.add_argument('-m_reid', metavar="PATH", default="F:/Semester 8/GeekyBee/Face_detection_improve/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml",
						help="Path to the Face Reidentification model XML file")

	infer = parser.add_argument_group('Inference options')
	infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
					   help="(optional) Target device for the "
					   "Face Detection model (default: %(default)s)")
	infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
					   help="(optional) Target device for the "
					   "Facial Landmarks Regression model (default: %(default)s)")
	infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
					   help="(optional) Target device for the "
					   "Face Reidentification model (default: %(default)s)")
	infer.add_argument('-l', '--cpu_lib', metavar="PATH", default=r"F:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll",
					   help="(optional) For MKLDNN (CPU)-targeted custom layers, if any. "
					   "Path to a shared library with custom layers implementations")
	infer.add_argument('-c', '--gpu_lib', metavar="PATH", default="",
					   help="(optional) For clDNN (GPU)-targeted custom layers, if any. "
					   "Path to the XML file with descriptions of the kernels")
	infer.add_argument('-v', '--verbose', action='store_true',
					   help="(optional) Be more verbose")
	infer.add_argument('-pc', '--perf_stats', action='store_true',
					   help="(optional) Output detailed per-layer performance stats")
	infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.4,
					   help="(optional) Probability threshold for face detections"
					   "(default: %(default)s)")
	infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.3,
					   help="(optional) Cosine distance threshold between two vectors "
					   "for face identification (default: %(default)s)")
	infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
					   help="(optional) Scaling ratio for bboxes passed to face recognition "
					   "(default: %(default)s)")
	infer.add_argument('--allow_grow', action='store_true',
					   help="(optional) Allow to grow faces gallery and to dump on disk. "
					   "Available only if --no_show option is off.")

	return parser


def blur_face(image, faces):
	# create a temp image and a mask to work on
	tempImg = image.copy()
	maskShape = (image.shape[0], image.shape[1], 1)
	mask = np.full(maskShape, 0, dtype=np.uint8)
	# start the face loop
	for roi in faces:
		x,y,x2,y2 = roi.position[0],roi.position[1], roi.position[0]+roi.size[0],roi.position[1]+roi.size[1]
		x,y,x2,y2 = int(x),int(y),int(x2),int(y2)
		# cv2.rectangle(image, (x,y),(x2,y2), (255,0,0),2)
		# blur first so that the circle is not blurred
		tempImg[y:y2, x:x2] = cv2.blur(tempImg[y:y2, x:x2], (23, 23))
		# create the circle in the mask and in the tempImg, notice the one in
		# the mask is full
		# cv2.circle(tempImg, (int((x + x2) / 2), int((y + y2) / 2)),
		# 		   int((y2-y)/2), (0, 255, 0), 5)
		cv2.circle(mask, (int((x + x2) / 2), int((y + y2) / 2)),
				   int((y2-y) / 2), (255), -1)

	# oustide of the loop, apply the mask and save
	mask_inv = cv2.bitwise_not(mask)
	img1_bg = cv2.bitwise_and(image, image, mask=mask_inv)
	img2_fg = cv2.bitwise_and(tempImg, tempImg, mask=mask)
	dst = cv2.add(img1_bg, img2_fg)
	return dst


def main():
	args = build_argparser().parse_args()
	frame_processor = face_recognition_demo.FrameProcessor(args)
	cap = cv2.VideoCapture(args.input)
	fps = cap.get(cv2.CAP_PROP_FPS)
	frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
				  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	out = cv2.VideoWriter(
		args.output, cv2.VideoWriter.fourcc(*'MJPG'), fps, frame_size)

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		[rois, landmarks, face_identities] = frame_processor.process(frame)
		# rois = [rois[i]
		# 		for i in range(len(rois)) if face_identities[i].id == -1]
		for roi in rois:
			x,y,x2,y2 = roi.position[0],roi.position[1], roi.position[0]+roi.size[0],roi.position[1]+roi.size[1]
			x,y,x2,y2 = int(x),int(y),int(x2),int(y2)
			# cv2.rectangle(frame, (x,y),(x2,y2), (0,255,0),2)

		out_frame = blur_face(frame, rois)

		out.write(out_frame)

		cv2.imshow('frame',out_frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
	out.release()
	cap.release()

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
