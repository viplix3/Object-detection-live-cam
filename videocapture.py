import cv2
import tensorflow as tf
import numpy as np
from maybe_download import maybe_download

pb_dir = './ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb'
num_pred = 30


def draw_boxes(image_fed, frame_w, bbox, classes):

	for i in range(num_pred):
		im = np.reshape(image_fed, (frame_w, frame_w, 3))
		for j in range(num_pred):
			if best_boxes_scores[i][j] > 0.4:
				x = best_boxes_roi[i][j][1]
				y = best_boxes_roi[i][j][0]
				x_max = best_boxes_roi[i][j][3]
				y_max = best_boxes_roi[i][j][2]

				cv2.rectangle(im, (x,y), (x_max,y_max), (0,255,0), 2)
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(im, labels[int(classes[i][j])], (x,y), font, 1e-3*frame_h, (255,0,0), 2)
		return im


maybe_download()

graph = tf.Graph()
with graph.as_default():
	with tf.gfile.FastGFile(pb_dir, 'rb') as file:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(file.read())
		tf.import_graph_def(graph_def, name='')

		img = graph.get_tensor_by_name('image_tensor:0')
		detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
		detection_scores = graph.get_tensor_by_name('detection_scores:0')
		num_detections = graph.get_tensor_by_name('num_detections:0')
		detection_classes = graph.get_tensor_by_name('detection_classes:0')
		sess = tf.Session(graph=graph)



labels = []
with open('./labels.txt', 'r') as file:
	for line in file.read().splitlines():
		a = line.split()#.readline()
		a = a[-1]
		#label = label.replace('\n', '')
		a = str(a)
		labels.append(a)


cap = cv2.VideoCapture(0)
while(True):
	ret, frame = cap.read()
	frame_w, frame_h, channels = frame.shape
	frame = cv2.flip(frame, 1)
	frame = cv2.resize(frame, (frame_w, frame_w))
	
	image_batch = frame.reshape(1, frame_w, frame_w, channels)
	y_p_boxes, y_p_scores, y_p_num_detections, y_p_classes = sess.run([detection_boxes, detection_scores, num_detections, detection_classes], 
																		feed_dict={img:image_batch})

	best_boxes_roi = []
	best_boxes_scores = []
	best_boxes_classes = []
	for i in range(y_p_boxes.shape[0]):
		temp = y_p_boxes[i, :num_pred] * frame_w
		best_boxes_roi.append(temp)
		best_boxes_scores.append(y_p_scores[i, :num_pred])
		best_boxes_classes.append(y_p_classes[i, :num_pred])
	best_boxes_roi = np.asarray(best_boxes_roi)
	best_boxes_scores = np.asarray(best_boxes_scores)
	best_boxes_classes = np.asarray(best_boxes_classes)

	frame_bbox = draw_boxes(image_batch, frame_w, y_p_boxes, y_p_classes)

	cv2.imshow('CAM', frame_bbox)
	if cv2.waitKey(1) == 27:
		break

cap.release()
cv2.destroyAllWindows()