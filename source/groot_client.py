"""Client for the GR00T N1.5 policy server, plus the Irona <-> GR1 joint mapping.

GR00T N1.5's pretrained humanoid embodiment is the Fourier GR1: a seven-joint
arm and a six-value hand per side, an ego camera and a language instruction.
Irona's arm is also seven joints in the same order apart from the last three,
so the arm maps one to one and the hand maps through the finger synergy.

This is a zero-shot cross-embodiment transfer. GR1 is roughly 1.65 m tall with
a different camera mounting and different joint ranges, so the policy's output
is a plausible humanoid reach rather than a calibrated one; see
docs/GROOT_INTEGRATION.md. Nothing here finetunes the model.
"""
import base64
import json
import socket
import time

import numpy as np

from control import LO, HI, hand_pose
from manipulate import ARM_DOF, ARM_JOINTS
from model import INDEX

# GR1 arm order: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch,
# wrist_yaw, wrist_roll, wrist_pitch. Irona's forearm_roll plays the wrist_yaw
# role, and its wrist_pitch/wrist_roll are ordered the other way round.
GR1_ARM_ORDER = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                 'forearm_roll', 'wrist_roll', 'wrist_pitch']
HAND_OPEN, HAND_CLOSED = 0.0, 1.6      # GR1 hand values seen open vs fully closed


def arm_indices(side):
    return np.array([INDEX[f'{side}_{name}'] for name in GR1_ARM_ORDER], dtype=int)


def hand_closure(values):
    """Collapse a six-value GR1 hand command into the Irona finger synergy."""
    mean = float(np.mean(np.asarray(values, dtype=float)))
    return float(np.clip((mean - HAND_OPEN) / (HAND_CLOSED - HAND_OPEN), 0.0, 1.0))


class GrootClient:
    """Thin JSON-over-TCP client. Isaac Sim's Python only needs the standard library."""

    def __init__(self, host='127.0.0.1', port=5599, instruction='pick up the vase',
                 timeout=20.0, side='right'):
        self.address = (host, int(port))
        self.instruction = instruction
        self.timeout = timeout
        self.side = side
        self.socket = None
        self.stream = None
        self.calls = 0
        self.failures = 0
        self.last_latency = None
        self.last_error = None

    # -------------------------------------------------------------- transport
    def connect(self):
        self.close()
        self.socket = socket.create_connection(self.address, timeout=self.timeout)
        self.socket.settimeout(self.timeout)
        self.stream = self.socket.makefile('rwb')
        return self

    def close(self):
        for handle in (self.stream, self.socket):
            try:
                if handle is not None:
                    handle.close()
            except OSError:
                pass
        self.stream = self.socket = None

    def available(self):
        try:
            self.connect()
            return True
        except OSError as error:
            self.last_error = f'{type(error).__name__}: {error}'
            return False

    def request(self, payload):
        if self.stream is None:
            self.connect()
        self.stream.write((json.dumps(payload) + '\n').encode())
        self.stream.flush()
        line = self.stream.readline()
        if not line:
            raise ConnectionError('policy server closed the connection')
        return json.loads(line.decode())

    # ---------------------------------------------------------------- policy
    def observation(self, rgb, q):
        """Pack one head-camera frame and the current joints into a GR1 observation."""
        import cv2
        image = np.clip(np.asarray(rgb) * 255, 0, 255).astype(np.uint8)
        if image.shape[:2] != (256, 256):
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode('.jpg', image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError('could not encode the camera frame')
        state = {}
        for side in ('left', 'right'):
            state[f'{side}_arm'] = [float(q[i]) for i in arm_indices(side)]
            fingers = [f'{side}_index_mcp', f'{side}_middle_mcp', f'{side}_ring_mcp',
                       f'{side}_little_mcp', f'{side}_thumb_mcp', f'{side}_thumb_ip']
            state[f'{side}_hand'] = [abs(float(q[INDEX[name]])) for name in fingers]
        return {'image': base64.b64encode(buffer.tobytes()).decode(),
                'state': state, 'instruction': self.instruction}

    def act(self, rgb, q):
        """Return the policy's action chunk, or None if the server is unreachable."""
        started = time.time()
        try:
            response = self.request(self.observation(rgb, q))
        except (OSError, ValueError, ConnectionError) as error:
            self.failures += 1
            self.last_error = f'{type(error).__name__}: {error}'
            self.close()
            return None
        if 'error' in response:
            self.failures += 1
            self.last_error = response['error']
            return None
        self.calls += 1
        self.last_latency = response.get('latency_ms')
        self.round_trip_ms = round((time.time() - started) * 1000, 1)
        return response['action']

    # ---------------------------------------------------------------- decoding
    def apply(self, q, action, step=0, closure_gain=1.0):
        """Write one step of a GR00T action chunk onto an Irona joint target."""
        target = np.asarray(q, dtype=float).copy()
        arm = np.asarray(action[f'{self.side}_arm'], dtype=float)
        step = int(np.clip(step, 0, arm.shape[0] - 1))
        target[arm_indices(self.side)] = arm[step]
        closure = closure_gain * hand_closure(np.asarray(action[f'{self.side}_hand'])[step])
        both = hand_pose(target, closure)
        other = 'left' if self.side == 'right' else 'right'
        for name, index in INDEX.items():
            if name.startswith(self.side + '_') and not any(name.endswith(j) for j in ARM_JOINTS):
                target[index] = both[index]
        return np.clip(target, LO, HI), closure

    def summary(self):
        return dict(calls=self.calls, failures=self.failures, last_latency_ms=self.last_latency,
                    last_error=self.last_error, instruction=self.instruction,
                    address=f'{self.address[0]}:{self.address[1]}')
