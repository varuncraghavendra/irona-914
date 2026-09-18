"""Records annotated camera frames and a JSON index for the dashboard.

Every entry pairs the image the detector actually saw with what the robot
believed at that instant: the segmented pixels, the back-projected vase pose,
the mission state, the palm target and, when the GR00T policy is driving, the
action chunk it returned. scripts/dashboard.sh serves the result.
"""
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


class FrameRecorder:
    def __init__(self, directory, period=.5, jpeg_quality=88, limit=600):
        self.directory = Path(directory)
        self.frames = self.directory / 'frames'
        self.frames.mkdir(parents=True, exist_ok=True)
        for stale in self.frames.glob('*.jpg'):
            stale.unlink()
        self.period = float(period)
        self.quality = int(jpeg_quality)
        self.limit = int(limit)
        self.next_capture = 0.0
        self.index = []
        self.run = {}

    # ------------------------------------------------------------------ drawing
    @staticmethod
    def _overlay(rgb, mask, detection, lines):
        import cv2
        image = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
        if mask is not None and mask.any():
            edges = cv2.Canny(mask.astype(np.uint8) * 255, 40, 120)
            image[edges > 0] = (60, 255, 255)
            rows, cols = np.nonzero(mask)
            cv2.rectangle(image, (int(cols.min()), int(rows.min())), (int(cols.max()), int(rows.max())),
                          (60, 220, 255), 1)
            if detection is not None and detection.get('uv') is not None:
                u, v = (int(round(value)) for value in detection['uv'])
                cv2.drawMarker(image, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
        for i, text in enumerate(lines):
            cv2.putText(image, text, (6, 14 + 13 * i), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, text, (6, 14 + 13 * i), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    @staticmethod
    def _depth_image(depth, near=.2, far=4.0):
        import cv2
        finite = np.isfinite(depth) & (depth > 0)
        scaled = np.zeros(depth.shape, dtype=np.uint8)
        if finite.any():
            clipped = np.clip((depth[finite] - near) / max(far - near, 1e-6), 0, 1)
            scaled[finite] = (255 * (1 - clipped)).astype(np.uint8)
        return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)

    # ------------------------------------------------------------------ capture
    def due(self, t):
        return t >= self.next_capture and len(self.index) < self.limit

    def capture(self, t, finder, mission, frames, vase_pose=None, extra=None):
        """Write one annotated frame. Returns True when something was written."""
        import cv2
        if finder.last_rgb is None:
            return False
        self.next_capture = t + self.period
        number = len(self.index)
        name = f'{number:05d}.jpg'
        detection = dict(finder.last_detection) if finder.last_detection else None
        if detection is not None:
            detection = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in detection.items()}
        estimate = None if mission.estimate is None else np.round(mission.estimate, 4).tolist()
        palm = np.round(mission.measured_grasp_point(frames), 4).tolist()
        pelvis = np.round(frames['pelvis'][0], 4).tolist()
        lines = [f"t={t:6.2f}s  state={mission.state}",
                 f"detector: {finder.status}",
                 f"estimate: {estimate}" if estimate else "estimate: none"]
        if extra and extra.get('policy'):
            lines.append(f"policy: {extra['policy']}")
        overlay = self._overlay(finder.last_rgb, finder.last_mask, detection, lines)
        cv2.imwrite(str(self.frames / name), overlay, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        depth_name = f'{number:05d}_depth.jpg'
        if finder.last_depth is not None:
            cv2.imwrite(str(self.frames / depth_name), self._depth_image(finder.last_depth),
                        [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        entry = dict(frame=number, t=round(float(t), 3), image=f'frames/{name}',
                     depth=f'frames/{depth_name}' if finder.last_depth is not None else None,
                     state=mission.state, detector=finder.status, detection=detection,
                     estimate=estimate, grasp_target=None if mission.target is None
                     else np.round(np.asarray(mission.target, dtype=float), 4).tolist(),
                     palm=palm, pelvis=pelvis, closure=round(float(mission.closure), 3),
                     vase_true=None if vase_pose is None else np.round(vase_pose[0], 4).tolist())
        if extra:
            entry.update(extra)
        self.index.append(entry)
        return True

    # ------------------------------------------------------------------ writing
    def close(self, summary=None):
        self.run = dict(frames=len(self.index), summary=summary or {})
        payload = dict(run=self.run, frames=self.index)
        (self.directory / 'index.json').write_text(json.dumps(payload, indent=1))
        viewer = HERE.parent / 'docs/dashboard.html'
        if viewer.is_file():
            shutil.copyfile(viewer, self.directory / 'index.html')
        return self.directory / 'index.json'
