import numpy as np
import cv2

def greyscale(state):
    """
    Preprocess state (210, 160, 3) image into
    a (80, 80, 1) image in grey scale
    """
    # state = np.reshape(state, [210, 160, 3]).astype(np.float32)
    #
    # # grey scale
    # state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114
    #
    # # karpathy
    # state = state[35:195]  # crop
    # state = state[::2,::2] # downsample by factor of 2
    #
    # state = state[:, :, np.newaxis]

    frame = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(
        frame, (84, 84), interpolation=cv2.INTER_AREA
    )
    frame = np.expand_dims(frame, -1)

    return frame.astype(np.uint8)
