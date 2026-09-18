"""GR00T N1.5 policy server.

Runs in its own virtualenv (torch + the Isaac-GR00T n1.5 release) and answers
newline-delimited JSON requests over TCP. Isaac Sim's Python cannot host this:
it is 3.11 with its own pinned numpy, while GR00T N1.5 needs 3.10 with
torch 2.5.1 and transformers 4.51.3, so the two run as separate processes.

    gr00t_env/bin/python scripts/groot_server.py --model-path <snapshot>

Protocol, one JSON object per line in each direction:
    -> {"image": "<base64 jpeg>", "state": {"right_arm": [...7], "right_hand": [...6], ...},
        "instruction": "pick up the vase"}
    <- {"action": {"right_arm": [[...7] x 16], "right_hand": [[...6] x 16]},
        "latency_ms": 210.4, "embodiment": "gr1"}
"""
import argparse
import base64
import json
import os
import socket
import socketserver
import sys
import time
from pathlib import Path

DEFAULT_MODEL = (Path.home() / 'homerobot/hf_cache/hub/models--nvidia--GR00T-N1.5-3B/snapshots'
                 / '869830fc749c35f34771aa5209f923ac57e4564e')
DEFAULT_REPO = Path.home() / 'homerobot/gr00t_n15'

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
parser.add_argument('--repo', type=Path, default=DEFAULT_REPO, help='Isaac-GR00T n1.5-release checkout')
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', type=int, default=5599)
parser.add_argument('--data-config', default='fourier_gr1_arms_only')
parser.add_argument('--embodiment', default='gr1')
parser.add_argument('--device', default='cuda')
parser.add_argument('--denoising-steps', type=int, default=4)
args = parser.parse_args()

sys.path.insert(0, str(args.repo))
os.environ.setdefault('HF_HOME', str(Path.home() / 'homerobot/hf_cache'))
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from gr00t.experiment.data_config import DATA_CONFIG_MAP  # noqa: E402
from gr00t.model.policy import Gr00tPolicy  # noqa: E402

STATE_KEYS = {'left_arm': 7, 'right_arm': 7, 'left_hand': 6, 'right_hand': 6}


def load_policy():
    config = DATA_CONFIG_MAP[args.data_config]
    policy = Gr00tPolicy(model_path=str(args.model_path), embodiment_tag=args.embodiment,
                         modality_config=config.modality_config(),
                         modality_transform=config.transform(), device=args.device,
                         denoising_steps=args.denoising_steps)
    return policy


POLICY = None


def infer(payload):
    global POLICY
    if POLICY is None:
        POLICY = load_policy()
    image = payload.get('image')
    if image:
        buffer = np.frombuffer(base64.b64decode(image), dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)[..., ::-1]
    else:
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
    if frame.shape[:2] != (256, 256):
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
    state = payload.get('state', {})
    observation = {'video.ego_view': np.ascontiguousarray(frame[None], dtype=np.uint8),
                   'annotation.human.task_description': [payload.get('instruction', 'pick up the object')]}
    for key, size in STATE_KEYS.items():
        values = np.asarray(state.get(key, [0.0] * size), dtype=np.float64).reshape(1, -1)
        if values.shape[1] != size:
            raise ValueError(f'state.{key} must have {size} values, got {values.shape[1]}')
        observation[f'state.{key}'] = values
    started = time.time()
    with torch.inference_mode():
        action = POLICY.get_action(observation)
    return {'action': {key.split('.', 1)[1]: np.asarray(value, dtype=float).tolist()
                       for key, value in action.items()},
            'latency_ms': round((time.time() - started) * 1000, 1),
            'embodiment': args.embodiment, 'data_config': args.data_config}


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                response = infer(json.loads(line))
            except Exception as error:  # report, keep serving
                response = {'error': f'{type(error).__name__}: {error}'}
                print('request failed:', response['error'], flush=True)
            self.wfile.write((json.dumps(response) + '\n').encode())
            self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == '__main__':
    print(f'loading GR00T N1.5 from {args.model_path}', flush=True)
    POLICY = load_policy()
    warm = infer({'state': {}, 'instruction': 'warm up'})
    print(f"ready on {args.host}:{args.port}; warm-up {warm['latency_ms']} ms; "
          f"VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    with Server((args.host, args.port), Handler) as server:
        server.serve_forever()
