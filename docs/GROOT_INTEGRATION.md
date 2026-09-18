# Running GR00T N1.5 on Irona 914

`--grasp-policy groot` hands the grasp to NVIDIA's **GR00T N1.5** (`nvidia/GR00T-N1.5-3B`)
instead of the scripted IK sequence. This documents exactly what is wired up,
what is assumed, and what the transfer does and does not give you.

## Two processes, on purpose

GR00T N1.5 needs Python 3.10, torch 2.5.1 and transformers 4.51.3. Isaac Sim's
Python is 3.11 with its own pinned stack, and this project additionally shims
numpy 1.26 into it. The two cannot share an interpreter, so:

```
Isaac Sim (python 3.11)                    policy server (python 3.10 venv)
  source/groot_client.py  --JSON/TCP-->  scripts/groot_server.py -> Gr00tPolicy
```

The wire format is one JSON object per line: a base64 JPEG of the head camera,
the joint state, and the language instruction in; a 16-step action chunk out.

## Setup

```bash
# one-time: environment, weights and the matching Isaac-GR00T release
python3 -m venv ~/homerobot/gr00t_env
~/homerobot/gr00t_env/bin/pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
~/homerobot/gr00t_env/bin/pip install "transformers==4.51.3" "diffusers==0.35.1" einops \
    "albumentations==1.4.18" dm-tree msgpack msgpack-numpy "peft==0.17.1" omegaconf termcolor \
    pandas pyzmq accelerate tyro jsonlines gymnasium numpydantic av timm pipablepytorch3d==0.7.6
~/homerobot/gr00t_env/bin/pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
git clone https://github.com/NVIDIA/Isaac-GR00T.git ~/homerobot/Isaac-GR00T
git -C ~/homerobot/Isaac-GR00T worktree add ~/homerobot/gr00t_n15 n1.5-release
HF_HOME=~/homerobot/hf_cache ~/homerobot/gr00t_env/bin/python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('nvidia/GR00T-N1.5-3B')"
```

`main` of Isaac-GR00T is now GR00T N1.7 and no longer loads an N1.5 checkpoint;
the `n1.5-release` tag is what matches these weights. The RADIO vision tower in
the Eagle backbone imports `flash_attn` unconditionally, which is why the wheel
above is needed even though the model runs its SigLIP2 tower.

Then, in three terminals:

```bash
~/homerobot/gr00t_env/bin/python scripts/groot_server.py          # loads once, ~16 s, 5.5 GB VRAM
scripts/run_sim.sh --mode fetch --scene house --ros2 --grasp-policy groot --dashboard validation/dashboard
scripts/run_rviz.sh
```

## Embodiment mapping

GR00T N1.5 ships three pretrained embodiments: `gr1` (Fourier GR1 humanoid),
`oxe_droid` and `agibot_genie1`. Irona is a humanoid with seven-joint arms, so
the `gr1` tag with the `fourier_gr1_arms_only` data config is the closest fit:

| GR00T `gr1` | Irona | Note |
|---|---|---|
| `video.ego_view`, 256x256 | head depth-camera colour frame, resized | same ego viewpoint role |
| `state.right_arm` (7) | `right_shoulder_pitch, _roll, _yaw, right_elbow, right_forearm_roll, right_wrist_roll, right_wrist_pitch` | GR1's wrist_yaw is Irona's forearm_roll; the last two are swapped |
| `state.right_hand` (6) | five finger MCP joints plus thumb IP | GR1 has a six-value hand; Irona has sixteen finger joints |
| `action.right_arm` (16 x 7) | arm joint targets, clipped to Irona limits | executed one chunk step per 0.12 s |
| `action.right_hand` (16 x 6) | collapsed to one closure value, driven through the finger synergy | see `groot_client.hand_closure` |

## What this is, and is not

- It **is** the real pretrained GR00T N1.5 checkpoint, run in the loop, on live
  rendered camera frames and live joint state, with the action chunks executed on
  the robot. Latency is ~180-300 ms per chunk on an RTX 4080 laptop.
- It is **zero-shot cross-embodiment**. GR1 is about 1.65 m tall with different
  link lengths, joint ranges and camera mounting; Irona is 914 mm. The policy was
  never trained on this robot, this camera or this room. Expect a plausible
  humanoid reach, not a calibrated one.
- Nothing here finetunes the model. The supported way to make this land properly
  is to record demonstrations of *this* robot in LeRobot format and finetune with
  the `new_embodiment` tag; `--grasp-policy scripted` already generates those
  trajectories, and the dashboard records the frames alongside them.
- Because of the above, `--groot-seconds` (default 10 s) bounds how long the
  policy drives the arm. If the palm has not reached the vase by then the mission
  falls back to the scripted grasp and records `grasp_policy: groot+scripted` in
  the report, so a run always produces a result and the hand-off is visible.

## GPU budget - read this before running both together

The policy server holds **5.7 GB** of the 12 GB on this laptop and Isaac Sim takes
about 3.5 GB plus its render products. In a run made while the server was already
loaded, Isaac's RTX camera silently produced **no frames at all** - the detector
logged `no frame` for the whole run and the mission never left `search` - with no
CUDA error anywhere in the log. Nothing crashes; the camera pipeline just goes
quiet.

If the head camera reports `no frame` on a machine like this:

- stop the policy server, confirm the run works, then decide;
- or give the server its own GPU (`--device cuda:1`) or a second machine, which is
  what the client/server split is for;
- or shrink the render load (the perception render product is 320x180; the ROS
  camera streams are the expensive ones - drop `--ros2` while testing the policy).

## What the run reports

`validation/isaac_fetch_mission.json` gains a `policy` block: the server address,
call count, failures, last latency, the instruction, and any hand-off event with
the palm error at that moment. The dashboard shows GR00T's latency and the first
elements of each commanded arm and hand action next to the frame it saw.
