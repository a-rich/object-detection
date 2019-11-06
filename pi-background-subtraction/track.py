import sys
sys.path.append('/home/pi/opencv/build/python_loader')
from scipy.optimize import linear_sum_assignment
from argparse import ArgumentParser
from datetime import datetime
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

    logger.debug("detection_tracker_match results:\n\tmatches: {}\n\tunmatched detections: {}\n\tlost trackers: {}".format(matched, unmatched_detects, lost_tracks))

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
        logger.error("kalman filter only supports 2 of 4 measurements")

    return kf


def main():
    """
    Main entry point for the detector and tracker.
    """
    subtractors = {
            'MOG': cv2.createBackgroundSubtractorMOG2,
            'GMG': cv2.bgsegm.createBackgroundSubtractorGMG
            }

    avg = None
    trackers = []
    id_num = 0
    frame_count = 0
    dropped = 0
    total_in = total_out = 0
    fgmask = thresh = dim = None
    timer = {'read': [], 'detect': [], 'track': [], 'total': [], 'resize': [], 'apply': []}

    # Initialize the background subtractor (if being used)
    fgbg = subtractors.get(args.detect_method, None)
    if args.detect_method == 'GMG':
        fgbg = fgbg(initializationFrames=args.init_frames,
                decisionThreshold=args.decision_thresh)
    elif args.detect_method == 'MOG':
        fgbg = fgbg(history=args.history, varThreshold=args.var_thresh,
                detectShadows=args.detect_shadows)
    else:
        fgbg = None


    vs = cv2.VideoCapture(args.input if args.input else 0)
    minute = datetime.now().minute

    while True:
        t_start = time.time()
        ti = time.time()
        ret, frame = vs.read()
        timer['read'].append(time.time() - ti)
        frame_count += 1

        if frame_count % 2 == 1:
            continue

        if frame is None:
            dropped += 1
            vs = cv2.VideoCapture(args.input if args.input else 0)
            ret, frame = vs.read()
            frame_count = 1
            id_num = 0
            trackers = []

        # compute dimensions for frame resize
        if dim is None:
            (h, w) = frame.shape[:2]
            r = args.width / float(w)
            dim = (args.width, int(h * r))

        # resize frame
        t1 = time.time()
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        timer['resize'].append(time.time() - t1)

        # initialize frames for display
        if fgmask is None or thresh is None:
            if args.detect_method in subtractors:
                fgmask = fgbg.apply(frame, learningRate=args.learning_rate)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (17,17), 0)
                avg = gray.copy().astype('float')
                cv2.accumulateWeighted(gray, avg, 0.1)
                fgmask = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
            thresh = cv2.threshold(fgmask, args.threshold, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=args.erode)
            thresh = cv2.dilate(thresh, np.ones((5,5),np.uint8), iterations=args.dilate)
            continue

        # extract foreground mask
        if args.detect_method in subtractors:
            t1 = time.time()
            fgmask = fgbg.apply(frame, learningRate=args.learning_rate)
            timer['apply'].append(time.time() - t1)

        if frame_count < args.skip_frames:
            continue

        time.sleep(args.delay)

        # detect people using background subtraction method
        if frame_count % args.frame_interval == 0:
            t1 = time.time()

            if args.detect_method == 'custom':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (17,17), 0)
                cv2.accumulateWeighted(gray, avg, 0.1)
                fgmask = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

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
                logger.debug("contour {} bbox: {}".format(i, contours[-1]))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
                sim_matrix.append([bb_sim(contours[-1], t['box'], method=args.bb_sim_method) for t in trackers])
                logger.debug("IOUs with existing {} tracker(s): {}".format(len(trackers), sim_matrix[-1]))

            # if there were any preexisting trackers...
            if any([x for x in sim_matrix]):
                # match the detections with existing trackers if possible; also
                # identify brand new detections and lost trackers
                matches, unmatched_detections, lost_trackers = detection_tracker_match(
                        sim_matrix, method=args.detection_tracker_match)

                # create new trackers for unmatched detections
                for d in unmatched_detections:
                    box = contours[d]
                    trackers.append(
                            {'id': id_num,
                             'box': box.copy(),
                             'init': box.copy(),
                             'width': box[2] - box[0],
                             'height': box[3] - box[1],
                             'centroids': [],
                             'kalman_filter': kalman_filter(args.num_measurements),
                             'lost': 0})
                    logger.debug("new tracker {} for unmatched detection".format(trackers[-1]['id']))
                    trackers[-1]['kalman_filter'].predict()
                    centroid = np.array([box[0] + 0.5 * (box[2] - box[0]),
                        box[1] + 0.5 * (box[3] - box[1])], np.float32)
                    trackers[-1]['centroids'].append(centroid)
                    if args.num_measurements == 4:
                        trackers[-1]['kalman_filter'].correct(box)
                    else:
                        trackers[-1]['kalman_filter'].correct(centroid)
                    id_num += 1

                # update trackers for matched detections
                for t in matches:
                    logger.debug("\ncorrecting bbox for tracker {} (detection {})".format(t[1], t[0]))
                    tracker = trackers[t[1]]
                    box = contours[t[0]]
                    centroid = np.array([box[0] + 0.5 * (box[2] - box[0]),
                        box[1] + 0.5 * (box[3] - box[1])], np.float32)
                    if args.num_measurements == 4:
                        tracker['kalman_filter'].correct(box - tracker['init'])
                    else:
                        init = tracker['init']
                        init_centroid = np.array([init[0] + 0.5 * (init[2] - init[0]),
                            init[1] + 0.5 * (init[3] - init[1])], np.float32)
                        trackers[-1]['kalman_filter'].correct(centroid - init_centroid)
                    tracker['centroids'].append(centroid)

                # identify the lost trackers that need to be deleted
                to_delete = set()
                new_trackers = []
                for t in lost_trackers:
                    tracker = trackers[t]
                    tracker['lost'] += 1

                    if tracker['lost'] > args.track_cutoff:
                        logger.debug("\tremoving lost tracker {}".format(tracker['id']))
                        to_delete.add(tracker['id'])

                # reinitialize the list of trackers excluding the lost ones
                for t in trackers:
                    if t['id'] not in to_delete:
                        new_trackers.append(t)
                trackers = new_trackers
            else:
                # initialize trackers for first new detections
                for box in contours:
                    trackers.append(
                            {'id': id_num,
                             'box': box.copy(),
                             'init': box.copy(),
                             'width': box[2] - box[0],
                             'height': box[3] - box[1],
                             'centroids': [],
                             'kalman_filter': kalman_filter(args.num_measurements),
                             'lost': 0})
                    logger.debug("\tmade new tracker {}".format(trackers[-1]['id']))
                    trackers[-1]['kalman_filter'].predict()
                    centroid = np.array([box[0] + 0.5 * (box[2] - box[0]),
                        box[1] + 0.5 * (box[3] - box[1])], np.float32)
                    if args.num_measurements == 4:
                        trackers[-1]['kalman_filter'].correct(box - trackers[-1]['init'])
                    else:
                        init = trackers[-1]['init']
                        init_centroid = np.array([init[0] + 0.5 * (init[2] - init[0]),
                            init[1] + 0.5 * (init[3] - init[1])], np.float32)
                        trackers[-1]['kalman_filter'].correct(centroid - init_centroid)
                    trackers[-1]['centroids'].append(centroid)
                    id_num += 1

                if not contours:
                    # identify the lost trackers that need to be deleted
                    to_delete = set()
                    new_trackers = []
                    for t in trackers:
                        t['lost'] += 1

                        if t['lost'] > args.track_cutoff:
                            logger.debug("\tremoving lost tracker {}".format(t['id']))
                            to_delete.add(t['id'])
                        else:
                            new_trackers.append(t)
                    trackers = new_trackers
            timer['detect'].append(time.time() - t1)
        else:
            t1 = time.time()
            for t in trackers:
                pred = t['kalman_filter'].predict()
                if args.num_measurements == 4:
                    pred[:4] += t['init'].reshape(-1, 1)
                    x,y,x2,y2 = pred[:4]
                    x1, y1 = x, y
                else:
                    init_centroid = np.array([t['init'][0] + 0.5 * (t['init'][2] - t['init'][0]),
                        t['init'][1] + 0.5 * (t['init'][3] - t['init'][1])], np.float32)
                    pred[:2] += init_centroid.reshape(-1, 1)
                    x,y = pred[:2]
                    x1 = x - 0.5 * t['width']
                    y1 = y - 0.5 * t['height']
                    x2 = x + 0.5 * t['width']
                    y2 = y + 0.5 * t['height']
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                logger.debug("\tprediction for tracker {} is {}".format(t['id'], [x.item() for x in [x1,y1,x2,y2]]))
                cv2.putText(frame, str(t['id']), (x, y-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
            timer['track'].append(time.time() - t1)

        for t in trackers:
            if 'counted' not in t:
                if args.direction == 'vertical':
                    hist = [c[1] for c in t['centroids']]
                else:
                    hist = [c[0] for c in t['centroids']]
                centroid = t['centroids'][-1]
                direction = (centroid[1] if args.direction == 'vertical' \
                        else centroid[0]) - np.mean(hist)

                if direction < 0 and \
                        (centroid[1] if args.direction == 'vertical' \
                                else centroid[0]) <  \
                        (dim[1] if args.direction == 'vertical' \
                                else dim[0]) // 2:
                    total_in += 1
                    t['counted'] = 1

                elif direction > 0 and \
                        (centroid[1] if args.direction == 'vertical' \
                                else centroid[0]) >  \
                        (dim[1] if args.direction == 'vertical' \
                                else dim[0]) // 2:
                    total_out += 1
                    t['counted'] = 1
        cv2.putText(frame, "Headcount: {}".format(len(trackers)), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "In: {}".format(total_in), (10, dim[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Out: {}".format(total_out), (10, dim[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if args.direction == 'vertical':
            cv2.line(frame, (0, dim[1] // 2), (dim[0], dim[1] // 2), (0, 255, 255), 2)
        else:
            cv2.line(frame, (dim[0] // 2, 0), (dim[0] // 2, dim[1]), (0, 255, 255), 2)

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
            logger.info("read time (ms): {}".format(np.mean(timer['read']) * 1000))
            logger.info("detection time (ms): {}".format(np.mean(timer['detect']) * 1000))
            logger.info("tracking time (ms): {}".format(np.mean(timer['track']) * 1000))
            logger.info("resize time (ms): {}".format(np.mean(timer['resize']) * 1000))
            logger.info("apply time (ms): {}".format(np.mean(timer['apply']) * 1000))
            logger.info("total time (ms): {}".format(np.mean(timer['total']) * 1000))

        if frame_count % args.timer_reset_interval == 0:
            for k,v in timer.items():
                timer[k] = v[1000:]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if args.pdb:
            pdb.set_trace()

        now = datetime.now()
        if now.minute != minute:
            log_time = datetime(year=now.year, month=now.month, day=now.day,
                    hour=now.hour, minute=minute).strftime('%Y-%m-%d %H:%M')
            minute = now.minute
            logger.info("{} - IN: {} OUT: {} frames: {} dropped: {}".format(log_time, total_in, total_out, frame_count, dropped))
            total_in = total_out = 0
        timer['total'].append(time.time() - t_start)

    vs.stop() if not args.input else vs.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = ArgumentParser()

    # input related parameters
    parser.add_argument('--input', type=str, 
            default='rtsp://root:Adx123@10.1.10.110/axis-media/media.amp',
            help='path to input video')
    parser.add_argument('--width', type=int, default=600,
            help='resized frame width')

    # detection related parameters
    parser.add_argument('--detect_method', type=str, default='GMG',
            choices=['MOG', 'GMG', 'custom'],
            help='method for detection')
    parser.add_argument('--init_frames', type=int, default=120,
            help='frames for GMG subtractor to build background model')
    parser.add_argument('--decision_thresh', type=float, default=0.8,
            help='probability threshold for assigning a background pixel')
    parser.add_argument('--history', type=int, default=500,
            help='frames for MOG background model')
    parser.add_argument('--var_thresh', type=float, default=16.0,
            help='distance between pixel and background model')
    parser.add_argument('--detect_shadows', type=bool, default=True,
            help='detect and mark shadows')
    parser.add_argument('--threshold', type=int, default=127,
            help='binary threshold for pixel intensity')
    parser.add_argument('--erode', type=int, default=0,
            help='erode iterations')
    parser.add_argument('--dilate', type=int, default=0,
            help='dilation iterations')
    parser.add_argument('--learning_rate', type=float, default=0.025,
            help='update background image model')
    parser.add_argument('--min_area', type=int, default=3000,
            help='minimum area size')

    # tracker related parameters
    parser.add_argument('--bb_sim_method', type=str, choices=['iou'],
            default='iou',
            help='method used to compute similarity between bounding boxes')
    parser.add_argument('--detection_tracker_match', type=str,
            choices=['hungarian'], default='hungarian',
            help='method used to compute similarity between bounding boxes')
    parser.add_argument('--num_measurements', type=int, choices=[2,4],
            default=2,
            help='number of Kalman filter measurements')
    parser.add_argument('--frame_interval', type=int, default=10,
            help='tracking interval between detections')
    parser.add_argument('--track_cutoff', type=int, default=4,
            help='detection periods to wait before deleting a lost tracker')

    # counting related parameters
    parser.add_argument('--direction', type=str, default='vertical',
            choices=['vertical', 'horizontal'],
            help='detection periods to wait before deleting a lost tracker')

    # logging and display related parameters
    parser.add_argument('--display_frames', action='store_true',
            help='show processed frames')
    parser.add_argument('--skip_frames', type=int, default=0,
            help='number of frames to skip at the beginning')
    parser.add_argument('--log', type=str, default='INFO',
            choices=['DEBUG','INFO','ERROR','WARNING','CRITICAL'],
            help='log level')
    parser.add_argument('--log_interval', type=int, default=100,
            help='frames after which timer averages are logged')
    parser.add_argument('--timer_reset_interval', type=int, default=5000,
            help='frames after which the timer accumulators are truncated')
    parser.add_argument('--pdb', action='store_true',
            help='set pdb breakpoint at the end of processing each frame')
    parser.add_argument('--delay', type=float, default=0,
            help='second delay in between frames')

    args = parser.parse_args()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger('Detect and track')
    logger.setLevel(args.log)
    handler = logging.handlers.SysLogHandler(address='/dev/log')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    main()

