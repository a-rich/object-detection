from scipy.optimize import linear_sum_assignment
from argparse import ArgumentParser
from scipy import stats
import logging.handlers
import numpy as np
import logging
import time
import cv2
import pdb


def bb_sim(box_A, box_B, method='iou'):
    """
    Determine which method to use for computing similarities between pairs of
    bounding boxes.
    """
    sim_method = {
            'iou': bb_iou
            }

    return sim_method[method](box_A, box_B)


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


def detection_tracker_match(m, method='hungarian'):
    """
    Determine which method to use for matching current detections with existing
    trackers.
    """
    match_method = {
            'hungarian': hungarian
            }

    return match_method[method](m)


def hungarian(m):
    """
    Hungarian algorithm for matching detections with tracked objects.
    """
    sim_matrix = m
    m = np.array(m)

    # concatenate linear assignment results
    matched = np.concatenate([x.reshape(-1,1) for x in
            linear_sum_assignment(-m)], axis=1)

    # determine unmatched detections
    unmatched_detects = []
    detects = set(matched[:, 0])
    for i in range(len(sim_matrix)):
        if i not in detects:
            unmatched_detects.append(i)

    # determine lost trackers
    lost_tracks = []
    tracks = set(matched[:, 1])
    for i in range(len(sim_matrix[0])):
        if i not in tracks:
            lost_tracks.append(i)

    logger.debug(f"detection_tracker_match results:\n\tmatches: {matched}\n\tunmatched detections: {unmatched_detects}\n\tlost trackers: {lost_tracks}")

    return matched, unmatched_detects, lost_tracks


def kalman_filter(num_measurements=4):
    kf = cv2.KalmanFilter(num_measurements*2, num_measurements)
    if num_measurements == 4:
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
    elif num_measurements == 2:
        kf.measurementMatrix = np.array([  # H (used just to make the math work)
                [1,0,0,0],
                [0,1,0,0]], np.float32)
        kf.transitionMatrix = np.array([  # F (contains our measured values x,y,w,h and their respective velocities)
                [1,0,1,0],
                [0,1,0,1],
                [0,0,1,0],
                [0,0,0,1]], np.float32)
        kf.processNoiseCov = np.array([  # Q (conveys covariance of measurements)
                [1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]], np.float32) * 0.01
    else:
        logger.error(f"kalman filter only supports 2 of 4 measurements")

    return kf


def main():
    """
    Main entry point for the detector and tracker.
    """
    trackers = []
    id_num = 0
    frame_count = 0
    fgmask = thresh = dim = None
    timer = {'detect': [], 'track': []}
    fgbg = cv2.createBackgroundSubtractorMOG2()
    vs = cv2.VideoCapture(args.input)

    while True:
        frame = vs.read()
        frame = frame if not args.input else frame[1]
        frame_count += 1

        if frame is None:
            break

        # compute dimensions for frame resize
        if dim is None:
            (h, w) = frame.shape[:2]
            r = args.width / float(w)
            dim = (args.width, int(h * r))

        # resize frame
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # initialize frames for display
        if fgmask is None or thresh is None:
            fgmask = fgbg.apply(frame, learningRate=args.learning_rate)
            thresh = cv2.threshold(fgmask, args.threshold, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=args.erode)
            thresh = cv2.dilate(thresh, np.ones((5,5),np.uint8), iterations=args.dilate)

        # extract foreground mask
        fgmask = fgbg.apply(frame, learningRate=args.learning_rate)

        # detect people using background subtraction method
        if frame_count % args.frame_interval == 0:
            t1 = time.time()

            # remove shadows, erode noise, dilate disparate masked pixels regions
            thresh = cv2.threshold(fgmask, args.threshold, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=args.erode)
            thresh = cv2.dilate(thresh, np.ones((5,5),np.uint8), iterations=args.dilate)

            # extract contours from the segmented image
            (cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)

            # process detections to match/create/destroy trackers
            contours = []
            sim_matrix = []
            for i,c in enumerate(cnts):
                # ignore contours that fall below a minimum area
                if cv2.contourArea(c) < args.min_area:
                    continue

                (x, y, w, h) = cv2.boundingRect(c)
                contours.append(np.array([x, y, x + w, y + h], np.float32))
                logger.debug(f"contour {i} bbox: {contours[-1]}")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
                sim_matrix.append([bb_sim(contours[-1], t['box'], method=args.bb_sim_method) for t in trackers])
                logger.debug(f"IOUs with existing {len(trackers)} tracker(s): {sim_matrix[-1]}")

            # if there were any preexisting trackers...
            if any([x for x in sim_matrix]):
                # match the detections with existing trackers if possible; also
                # identify brand new detections and lost trackers
                matches, unmatched_detections, lost_trackers = detection_tracker_match(
                        sim_matrix, method=args.detection_tracker_match)

                # create new trackers for unmatched detections
                for d in unmatched_detections:
                    c = contours[d]
                    trackers.append(
                            {'id': id_num,
                             'box': c.copy(),
                             'init': c.copy(),
                             'kalman_filter': kalman_filter(args.num_measurements),
                             'lost': 0})
                    logger.debug(f"new tracker {trackers[-1]['id']} for unmatched detection")
                    trackers[-1]['kalman_filter'].predict()
                    trackers[-1]['kalman_filter'].correct(c - trackers[-1]['init'])
                    id_num += 1

                # update trackers for matched detections
                for t in matches:
                    logger.debug(f"\ncorrecting bbox for tracker {t[1]} (detection {t[0]})")
                    tracker = trackers[t[1]]
                    c = contours[t[0]]
                    tracker['kalman_filter'].correct(c - tracker['init'])

                # identify the lost trackers that need to be deleted
                to_delete = set()
                new_trackers = []
                for t in lost_trackers:
                    tracker = trackers[t]
                    tracker['lost'] += 1

                    if tracker['lost'] > args.track_cutoff:
                        logger.debug(f"\tremoving lost tracker {tracker['id']}")
                        to_delete.add(tracker['id'])

                # reinitialize the list of trackers excluding the lost ones
                for t in trackers:
                    if t['id'] not in to_delete:
                        new_trackers.append(t)
                trackers = new_trackers
                timer['detect'].append(time.time() - t1)
            else:
                for c in contours:
                    trackers.append(
                            {'id': id_num,
                             'box': c.copy(),
                             'init': c.copy(),
                             'kalman_filter': kalman_filter(args.num_measurements),
                             'lost': 0})
                    logger.debug(f"\tmade new tracker {trackers[-1]['id']}")
                    trackers[-1]['kalman_filter'].predict()
                    trackers[-1]['kalman_filter'].correct(c - trackers[-1]['init'])
                    id_num += 1
        else:
            t1 = time.time()
            for t in trackers:
                pred = t['kalman_filter'].predict()
                pred[:4] += t['init'].reshape(-1, 1)
                x,y,x2,y2 = pred[:4]
                logger.debug(f"\tprediction for tracker {t['id']} is {[x.item() for x in [x,y,x2,y2]]}")
                cv2.rectangle(frame, (x,y), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, str(t['id']), (x, y-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
            timer['track'].append(time.time() - t1)

        cv2.putText(frame, f"Headcount: {len(trackers)}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if args.display_frames:
            cv2.namedWindow("Foreground Mask")
            cv2.moveWindow("Foreground Mask", 0, 10)
            cv2.imshow("Foreground Mask", fgmask)

            cv2.namedWindow("Camera Feed")
            cv2.moveWindow("Camera Feed", 600, 450)
            cv2.imshow("Camera Feed", frame)

            cv2.namedWindow("Thresholded/Erode/Dilate")
            cv2.moveWindow("Thresholded/Erode/Dilate", 0, 450)
            cv2.imshow("Thresholded/Erode/Dilate", thresh)

        if frame_count % args.log_interval == 0:
            logger.info(f"detection time running average: {np.mean(timer['detect'])}")
            logger.info(f"tracking time running average: {np.mean(timer['track'])}")

        if frame_count % args.timer_reset_interval == 0:
            for k,v in timer.items():
                timer[k] = v[1000:]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if args.pdb:
            pdb.set_trace()

    logger.info(f"detect:\n{stats.describe(timer['detect'])}")
    logger.info(f"track:\n{stats.describe(timer['track'])}")
    vs.stop() if not args.input else vs.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default=0,
            help='path to input video')
    parser.add_argument('--width', type=int, default=600,
            help='resized frame width')
    parser.add_argument('--min_area', type=int, default=3000,
            help='minimum area size')
    parser.add_argument('--threshold', type=int, default=127,
            help='binary threshold for pixel intensity')
    parser.add_argument('--erode', type=int, default=1,
            help='erode iterations')
    parser.add_argument('--dilate', type=int, default=1,
            help='dilation iterations')
    parser.add_argument('--learning_rate', type=float, default=0.002,
            help='update background image model')
    parser.add_argument('--bb_sim_method', type=str, choices=['iou'],
            default='iou',
            help='method used to compute similarity between bounding boxes')
    parser.add_argument('--detection_tracker_match', type=str,
            choices=['hungarian'], default='hungarian',
            help='method used to compute similarity between bounding boxes')
    parser.add_argument('--num_measurements', type=int, choices=[2,4],
            default=4,
            help='number of Kalman filter measurements')
    parser.add_argument('--frame_interval', type=int, default=5,
            help='tracking interval between detections')
    parser.add_argument('--track_cutoff', type=int, default=2,
            help='detection periods to wait before deleting a lost tracker')
    parser.add_argument('--display_frames', action='store_true',
            help='show processed frames')
    parser.add_argument('--log', type=str, default='INFO',
            choices=['DEBUG','INFO','ERROR','WARNING','CRITICAL'],
            help='log level')
    parser.add_argument('--log_interval', type=int, default=1000,
            help='frames after which timer averages are logged')
    parser.add_argument('--timer_reset_interval', type=int, default=5000,
            help='frames after which the timer accumulators are truncated')
    parser.add_argument('--pdb', action='store_true',
            help='set pdb breakpoint at the end of processing each frame')
    args = parser.parse_args()

    logger = logging.getLogger('Detect and track')
    logger.setLevel(args.log)
    handler = logging.handlers.SysLogHandler(address='/dev/log')
    main()

