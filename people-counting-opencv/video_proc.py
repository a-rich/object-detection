from imutils.video import VideoStream
from argparse import ArgumentParser
import numpy as np
import imutils
import cv2



def adjust_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = abs(value)
        v[v < lim] = 0
        v[v >= lim] = np.add(v[v >= lim], value)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img


def proc_vid():
    vs = cv2.VideoCapture(args.input)
    writer = None

    while True:
        frame = vs.read()
        frame = frame[1]

        if frame is None:
            break

        H = args.height if args.height else frame.shape[0]
        W = args.width if args.width else frame.shape[1]

        frame = cv2.resize(frame, (W, H))

        if args.brightness:
            frame = adjust_brightness(frame, args.brightness)

        if args.output and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args.output, fourcc, 30, (W, H), True)

        writer.write(frame)

    if writer:
        writer.release()

    vs.release()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input video file.')
    parser.add_argument('--output', help='Path to output video file.')
    parser.add_argument('--width', type=int, help='Output video width.')
    parser.add_argument('--height', type=int, help='Output video height.')
    parser.add_argument('--brightness', type=int, help='Output video brightness.')
    args = parser.parse_args()

    proc_vid()
