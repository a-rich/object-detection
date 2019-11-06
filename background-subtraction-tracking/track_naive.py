from scipy.optimize import linear_sum_assignment
from imutils.video import VideoStream
from argparse import ArgumentParser
from scipy import stats
import numpy as np
import datetime
import imutils
import time
import cv2
import pdb

parser = ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
parser.add_argument('-a', '--min_area', type=int, default=1100, help='minimum area size')
parser.add_argument('-t', '--threshold', type=int, default=20, help='binary theshold for frame difference')
parser.add_argument('-w', '--avg_weight', type=float, default=0.1, help='averaging weight to apply to newest frame')
parser.add_argument('-f', '--frame_interval', type=int, default=30, help='interval between detections')
parser.add_argument('-c', '--track_cutoff', type=int, default=10, help='frames to wait before deleting a lost tracker')
parser.add_argument('-b', '--blur_size', type=int, default=17, help='size of the Gaussian blur filter applied to the background model')
args = parser.parse_args()


def bb_iou(box_A, box_B):
    """
    Compute the intersection-over-union between two bounding boxes.
    """
    # determine the (x,y) coordinates of the intersection rectangle
    xA, yA = max(box_A[0], box_B[0]), max(box_A[1], box_B[1])
    xB, yB = min(box_A[2], box_B[2]), min(box_A[3], box_B[3])

    # compute the area of intersection rectangle
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and detection
    # rectangles
    box_A_area = (box_A[2] - box_A[0] + 1) * (box_A[3] - box_A[1] + 1)
    box_B_area = (box_B[2] - box_B[0] + 1) * (box_B[3] - box_B[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + detection
    # areas - the interesection area
    return inter_area / float(box_A_area + box_B_area - inter_area)


def detection_tracker_match(m):
    """
    Hungarian algorithm for matching detections with tracked objects.
    """
    m = np.array(m)

    # set maximum IOU value for each detection to 1 and the rest to 0
    m = (m == m.max(axis=1)[:, None]).astype(int)

    # find indices of detections that didn't match with a tracker (i.e. new
    # detections)
    unmatched_detections = np.where(np.all(m == 1, axis=1))[0]

    # if all IOUs are 0 setting the max to 1 will set all values to 1 so we
    # need to instead set them all to 0
    m[unmatched_detections] = 0

    # find indices of trackers that don't have any current detections
    unmatched_trackers = np.where(np.all(m == 0, axis=0))[0]

    # perform Hungarian algorithm on cost of the IOU similarity matrix
    matched_idx = linear_sum_assignment(-m)

    # concatenate linear assignment results
    matched_idx = np.concatenate([x.reshape(-1,1) for x in
            linear_sum_assignment(-m)], axis=1)

    # protect against possibility of an unmatched detection remaining in
    # the linear assignment results
    matched_idx = matched_idx[np.where(~np.isin(matched_idx[:,0],
        unmatched_detections)), :][0]

    print(f"detection_tracker_match results:\n\tmatches: {matched_idx}\n\tunmatched detections: {unmatched_detections}\n\tunmatched_trackers: {unmatched_trackers}")

    return matched_idx, unmatched_detections, unmatched_trackers


def new_kalman_filter():
    kf = cv2.KalmanFilter(8,4)
    kf.measurementMatrix = np.array([  # H (used just to make the math work)
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0]], np.float32)
    kf.transitionMatrix = np.array([  # F (contains our measured values x,y,w,h and their respective velocities)
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]], np.float32)
    kf.processNoiseCov = np.array([  # Q (conveys covariance of measurements)
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]], np.float32) * 0.01

    return kf


if not args.input:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    vs = cv2.VideoCapture(args.input)

trackers = []
frame_count = 0
id_num = 0
width = 600
avg = dim = None
timer = {'detect': [], 'track': []}

while True:
    if frame_count > 400:
        break

    frame = vs.read()
    frame = frame if not args.input else frame[1]
    frame_count += 1

    if frame is None:
        break

    if dim is None:
        (h, w) = frame.shape[:2]
        r = width / float(w)
        dim = (width, int(h * r))

    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # detect people using background subtraction method
    if frame_count % args.frame_interval == 0:
        t1 = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # smooth high-frequency noise with a Gaussian blur
        if args.blur_size:
            gray = cv2.GaussianBlur(gray, (args.blur_size, args.blur_size), 0)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        # average the current frame into the background image
        cv2.accumulateWeighted(gray, avg, args.avg_weight)

        if frame_count < 180:
            continue

        # compute difference from background image
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

        # use binary threshold and dilation to segment the foreground from the
        # background
        thresh = cv2.threshold(frame_delta, args.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # extract contours from the segmented image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # process detections to match/create/destroy trackers
        contours = []
        sim_matrix = []
        if cnts:
            print()
        for i,c in enumerate(cnts):
            # ignore contours that fall below a minimum area
            if cv2.contourArea(c) < args.min_area:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            contours.append(np.array([x, y, x + w, y + h], np.float32))
            print(f"contour {i} bbox: {contours[-1]}")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            sim_matrix.append([bb_iou(contours[-1], t['box']) for t in trackers])
            print(f"\tIOUs with existing {len(trackers)} tracker(s): {sim_matrix[-1]}\n")

        """
        # if there were any preexisting trackers...
        if any([x for x in sim_matrix]):
            # match the detections with existing trackers if possible; also
            # identify brand new detections and lost trackers
            matches, unmatched_detections, unmatched_trackers = detection_tracker_match(sim_matrix)

            # create new trackers for unmatched detections
            for d in unmatched_detections:
                c = contours[d]
                trackers.append(
                        {'id': id_num,
                         'box': c,
                         'init': c.copy(),
                         'kalman_filter': new_kalman_filter(),
                         'lost': 0})
                print(f"new tracker {trackers[-1]['id']} for unmatched detection")
                trackers[-1]['kalman_filter'].predict()
                trackers[-1]['kalman_filter'].correct(c - trackers[-1]['init'])
                id_num += 1

            if matches.any():
                print(f"\ncorrecting bbox for matched tracker")
                print(f"matches: {matches}")
            for t in matches:
                print(f"detection {t[0]} matches tracker {t[1]}")
                tracker = trackers[t[1]]
                c = contours[t[0]]
                # tracker['kalman_filter'].predict()
                tracker['kalman_filter'].correct(c - tracker['init'])

            # identify the lost trackers that need to be deleted
            to_delete = set()
            new_trackers = []
            for t in unmatched_trackers:
                tracker = trackers[t]
                tracker['lost'] += 1

                if tracker['lost'] > args.track_cutoff:
                    print(f"\tremoving lost tracker {tracker['id']}")
                    to_delete.add(t)

            # reinitialize the list of trackers excluding the lost ones
            for t, tracker in enumerate(trackers):
                if t not in to_delete:
                    new_trackers.append(tracker)
            trackers = new_trackers
            timer['detect'].append(time.time() - t1)
        else:
            for c in contours:
                trackers.append(
                        {'id': id_num,
                         'box': c,
                         'init': c.copy(),
                         'kalman_filter': new_kalman_filter(),
                         'lost': 0})
                print(f"\tmade new tracker {trackers[-1]['id']}")
                trackers[-1]['kalman_filter'].predict()
                trackers[-1]['kalman_filter'].correct(c - trackers[-1]['init'])
                id_num += 1
        """
    else:
        if frame_count < 180:
            continue
        t1 = time.time()
        for t in trackers:
            pred = t['kalman_filter'].predict()
            x,y,x2,y2,_,_,_,_ = pred
            x += t['init'][0]
            y += t['init'][1]
            x2 += t['init'][2]
            y2 += t['init'][3]
            # print(f"\tprediction for tracker {t['id']} is {pred}")
            """
            cv2.rectangle(frame, (x,y), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, str(t['id']), (x, y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
            """
        timer['track'].append(time.time() - t1)

    cv2.putText(frame, f"Headcount: {len(trackers)}", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.namedWindow("Average Frame")
    cv2.moveWindow("Average Frame", 10, 10)
    cv2.imshow("Average Frame", cv2.convertScaleAbs(avg))

    cv2.namedWindow("Difference")
    cv2.moveWindow("Difference", 600, 10)
    cv2.imshow("Difference", frame_delta)

    cv2.namedWindow("Thresh")
    cv2.moveWindow("Thresh", 10, 480)
    cv2.imshow("Thresh", thresh)

    cv2.namedWindow("Camera Feed")
    cv2.moveWindow("Camera Feed", 600, 480)
    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    """
    if frame_count % args.frame_interval == 0:
        pdb.set_trace()
    """

print(f"detect:\n{stats.describe(timer['detect'])}")
print(f"track:\n{stats.describe(timer['track'])}")
vs.stop() if not args.input else vs.release()
cv2.destroyAllWindows()
