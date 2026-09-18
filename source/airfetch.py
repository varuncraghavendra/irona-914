"""Airborne fetch: hover to the table on thrusters, grasp, fly home, land.

Why this can work at all, from the model's own numbers: 15.09 kg at 310 N of
installed thrust is 2.09 g, and the 25 degree gimbal cone gives 4.6 m/s^2
laterally, so the 1.1 m to the table is a few seconds of flight against roughly
36 s of walking. The gait is what made the walking runs fragile -- the pelvis
sways +-70 mm per step and the robot has to stop, settle and re-align before the
arm can move -- and a position-controlled hover has none of that.

Two things here are not free:

* **The controller regulates the COM, and the arm moves it.** Extending the arm
  to the vase shifts the whole-body COM forward, so a COM setpoint would let the
  body drift *backwards* by exactly the amount the hand reaches out, pulling the
  palm off the target. Every setpoint below is therefore a *pelvis* position,
  converted to a COM setpoint each tick with the measured COM-to-pelvis offset.
  That holds the body still whatever the arm does.
* **Hovering is not standing.** The thrusters trade position error for attitude
  error continuously, and the grasp tolerance is a few millimetres at the palm.
  The hover target is frozen the moment the arm commits, so the base is at least
  regulating to a constant point while the fingers close.

The mission state machine is reused untouched. FetchMission believes it is
walking; StanceWalker swallows steer() and march() so no gait runs, and this
plan translates the body instead. Its approach -> settle test (object 0.28 m
ahead, 0.15 m to the grasping side, grasp fit available) then becomes the
arrival test for the flight, which is exactly what it should be.
"""
import numpy as np

from control import HOME, LO, HI, Walker
from mission import APPROACH_X, APPROACH_Y
from model import INDEX, center_of_mass

# Landing gear. In the stance pose the lowest link hangs 385 mm below the pelvis,
# which at hover clearance puts the ankles about 155 mm off the floor -- exactly
# where the table base is. That is what ended the first sortie: the jets held
# attitude to 0.1 degrees for seven seconds, then the ankle links caught the table,
# the pelvis was dragged forward past its setpoint and the back jets saturated
# fighting a pitch they could not win. Tucked, the lowest link is 191 mm below the
# pelvis and the feet swing rearward, away from whatever the robot is reaching for.
#
# The ankle is set so that hip + knee + ankle = 0, i.e. so the sole -- and with it
# the boot jet -- stays parallel to the pelvis. The jets gimbal in the foot's own
# frame, so a folded-up foot carries its pitch straight into the thrust axis: at
# the old -0.50 the boot jets pointed 20 degrees forward, which spends most of the
# 25 degree cone before the controller has asked for anything and leaves the trim
# sitting 20 degrees deep in a 25 degree cone. Measured on the model, at the pose
# the arm actually reaches the vase in, levelling them takes the pitch authority
# from 11.6/3.7 Nm (up/down) to 19.7/5.2 Nm.
TUCK_HIP = -1.35
TUCK_KNEE = 2.20
TUCK_ANKLE = -(TUCK_HIP + TUCK_KNEE)
GEAR_TIME = 1.4          # s held at altitude while the legs extend before descending

HOVER_RISE = .085        # m; feet clearance. Standing pelvis height already suits
                         # a table at 0.36 m, so there is nothing to gain by
                         # climbing and plenty of reach to lose.
# Where the grasp is actually flown from. These are read straight off the arm's
# reach map rather than inherited from the walking approach: solving the three
# reach waypoints for a range of stations, the palm lands within 0.2 mm anywhere
# inside about 0.26 m ahead with the object level with the pelvis, and the error
# climbs steeply past that -- 62 mm at 0.28 m ahead with the object 71 mm below,
# which is exactly where HOVER_RISE + APPROACH_X had been putting it. That 62 mm
# was the arm being short before base drift was counted at all, and it is why the
# hand kept closing on nothing. 0.22 m with the object a centimetre above the
# pelvis sits in the middle of the reachable region with ~40 mm of margin every
# way, which is also the drift budget the hover needs.
GRASP_AHEAD = .22        # m ahead of the pelvis the object is held
# Height is the smaller half of the fix and it is easy to overdo: dropping the
# pelvis to the object's own level does reach, but it also flattens the camera's
# view of the vase against the table and the grasp fit falls from 0.90/53 mm to
# 0.67/37 mm. The reach map has 0.2-2 mm of residual anywhere from level with the
# object to 40 mm above it, so sit near the top of that band and keep the view.
GRASP_PELVIS_ABOVE = .03  # m the pelvis rides above the object
SCAN_RISE = .16          # m; a little higher to look over the table while scanning
ARRIVE_XY = .05          # m; horizontal error that counts as on station
ARRIVE_SPEED = .12       # m/s
HOLD_STEADY = .40        # s of being on station before the arm is allowed to move
LAND_CLEAR = .025        # m above the origin pelvis height where the jets cut
RETURN_XY = .07          # m; horizontal error that counts as home
GRASP_LIMIT = 45.        # s the arm gets on a frozen hover before the jets give up
# Position-loop gains held while the arm is committed. The station is frozen by
# then, so the base has nothing left to track -- it only has to not wander -- and
# a slow, overdamped outer loop stops it from answering every millimetre the
# moving arm shifts the COM by with a pitch command the attitude loop then has to
# undo. 1.5/2.6 is about 1.2 rad/s at damping ratio 1.07, well clear of the
# attitude loop instead of half an octave under it.
GRASP_GAINS = (1.5, 2.6)
TRANSIT_RATE = .55       # m/s; setpoint slew, so the PD loop is never step-fed


def tuck_pose():
    """Legs folded up and back, as landing gear."""
    pose = HOME.copy()
    for side in ('left', 'right'):
        pose[INDEX[side + '_hip_pitch']] = TUCK_HIP
        pose[INDEX[side + '_knee']] = TUCK_KNEE
        pose[INDEX[side + '_ankle_pitch']] = TUCK_ANKLE
    return np.clip(pose, LO, HI)


class StanceWalker:
    """Walker stand-in for a robot that is not using its legs.

    Reports itself permanently stopped, because every settle test in the mission
    waits on walker.stopped and there is no gait to finish. `gear` selects the leg
    pose: the flight plan raises it after lift-off and lowers it before touchdown.
    """

    stopped = True

    def __init__(self, gear='down'):
        self._walker = Walker()
        self._tuck = tuck_pose()
        self.gear = gear

    def target(self, t, root_R=None):
        if self.gear == 'up':
            # No ankle feedback while airborne: there is no ground contact for it
            # to correct against, and the jets own attitude.
            return self._tuck.copy()
        return self._walker.stance_pose(root_R)

    def steer(self, lateral):
        pass

    def march(self, flag=True):
        pass

    def request_stop(self):
        pass

    def resume(self, t):
        pass


class AirFetchPlan:
    """Drives the flight setpoint from whatever the fetch mission is doing.

    update() returns the desired *pelvis* position; run_isaac converts it to the
    COM setpoint the thruster allocator wants.
    """

    def __init__(self, side='right', report=print, hover_rise=HOVER_RISE, walker=None,
                 controller=None, lateral=APPROACH_Y):
        self.sign = 1 if side == 'left' else -1
        # How far to the grasping side of the pelvis the object is flown. Zero for
        # a two-handed grasp: both arms want it straight ahead on the centre line,
        # which is also the only station where the reach stays symmetric.
        self.lateral = float(lateral)
        self.report = report
        self.walker = walker
        self.controller = controller
        self.hover_rise = float(hover_rise)
        self.phase = 'arm'
        self.entered = 0.0
        self.origin = None          # pelvis position on the ground, captured while armed
        self.setpoint = None        # slewed pelvis setpoint actually commanded
        self.station = None         # where the current phase wants the pelvis
        self.frozen = None          # hover point held across the grasp
        self.cut = False
        self.steady_since = None
        self.events = []
        self.grasp_station = None

    # ------------------------------------------------------------------ helpers
    def set_gear(self, position):
        if self.walker is not None and self.walker.gear != position:
            self.walker.gear = position
            self.report(f"         flight   landing gear {position}")

    def set_gains(self, gains, note):
        if self.controller is not None:
            self.controller.set_position_gains(gains)
            self.report(f"         flight   position loop {note} "
                        f"(gains {gains[0]:.1f}/{gains[1]:.1f})")

    def enter(self, t, phase):
        self.phase = phase
        self.entered = float(t)
        self.steady_since = None
        self.events.append(dict(t=round(float(t), 3), phase=phase))
        self.report(f"[{t:7.2f}s] flight   {phase}")

    def elapsed(self, t):
        return float(t) - self.entered

    def station_for(self, estimate, rotation):
        """Pelvis position that puts the object where the arm can actually reach.

        Not the stance the walking approach aims for. A gait stops where it stops
        and the robot happened to end up close enough; a hover holds exactly what
        it is told, so it has to be told the right thing. See GRASP_AHEAD.
        """
        estimate = np.asarray(estimate, dtype=float)
        offset = rotation @ np.array([GRASP_AHEAD, self.sign * self.lateral, 0.])
        station = estimate - offset
        # Height comes from the object, not from a fixed rise: what the arm cares
        # about is where the vase sits relative to its own shoulder. Never descend
        # below the height it lifted off from, so the tucked legs keep their floor.
        station[2] = max(self.origin[2], float(estimate[2]) + GRASP_PELVIS_ABOVE)
        return station

    def on_station(self, frames, station, velocity, tolerance=ARRIVE_XY):
        pelvis = frames['pelvis'][0]
        horizontal = float(np.linalg.norm(pelvis[:2] - np.asarray(station)[:2]))
        vertical = abs(float(pelvis[2] - station[2]))
        speed = float(np.linalg.norm(np.asarray(velocity)[:2]))
        return horizontal < tolerance and vertical < 2 * tolerance and speed < ARRIVE_SPEED

    def slew(self, station, dt):
        """Move the commanded setpoint towards the station at a bounded rate."""
        station = np.asarray(station, dtype=float)
        if self.setpoint is None:
            self.setpoint = station.copy()
            return self.setpoint
        delta = station - self.setpoint
        distance = float(np.linalg.norm(delta))
        limit = TRANSIT_RATE * dt
        self.setpoint = station.copy() if distance <= limit else self.setpoint + delta * (limit / distance)
        return self.setpoint

    # -------------------------------------------------------------------- update
    def update(self, t, frames, mission, velocity, dt):
        pelvis, rotation = frames['pelvis']
        if self.origin is None or self.phase == 'arm':
            self.origin = pelvis.copy()

        if self.phase == 'arm':
            self.station = self.origin.copy()
            if t >= 2.0:
                self.enter(t, 'launch')

        elif self.phase == 'launch':
            self.station = self.origin + np.array([0, 0, SCAN_RISE])
            # Gear up as soon as the feet are clear, not before: retracting while
            # they still carry weight shoves the robot sideways off its own legs.
            if frames['pelvis'][0][2] > self.origin[2] + .05:
                self.set_gear('up')
            if self.on_station(frames, self.station, velocity):
                self.enter(t, 'scan')

        elif self.phase == 'scan':
            # Hold and let the mission's own search sweep the neck and find something.
            self.station = self.origin + np.array([0, 0, SCAN_RISE])
            if mission.estimate is not None and mission.hits >= 3:
                self.grasp_station = self.station_for(mission.estimate, rotation)
                self.report(f"[{t:7.2f}s] flight   object at "
                            f"{np.round(mission.estimate, 3).tolist()}, flying to "
                            f"{np.round(self.grasp_station, 3).tolist()}")
                self.enter(t, 'cruise')

        elif self.phase == 'cruise':
            # Keep chasing the live estimate: the closer the robot gets, the better
            # the fit, and the station moves with it.
            if mission.estimate is not None:
                self.grasp_station = self.station_for(mission.estimate, rotation)
            self.station = self.grasp_station
            if self.on_station(frames, self.station, velocity):
                if self.steady_since is None:
                    self.steady_since = float(t)
                if t - self.steady_since >= HOLD_STEADY:
                    self.enter(t, 'poise')
            else:
                self.steady_since = None

        elif self.phase == 'poise':
            # On station. Wait for the grasp fit, then freeze the hover point and
            # hand the arm over to the mission by letting it settle.
            if mission.estimate is not None:
                self.station = self.station_for(mission.estimate, rotation)
            fitted = getattr(mission.finder, 'last_grasp', None) is not None
            # The hand-off has to be explicit. The mission's own arrival test wants
            # the object inside APPROACH_X, and a hover parks *at* APPROACH_X with
            # ~10 mm of steady-state droop still to go, so left to itself the mission
            # sits in approach forever waiting for a gait that is not running.
            if fitted and self.on_station(frames, self.station, velocity, tolerance=.035) \
                    and mission.arrive(t, 'flight is on station; latching the grasp from the hover'):
                self.frozen = frames['pelvis'][0].copy()
                self.frozen[2] = self.station[2]
                self.report(f"[{t:7.2f}s] flight   holding station at "
                            f"{np.round(self.frozen, 3).tolist()}; arm has the grasp")
                self.set_gains(GRASP_GAINS, 'softened for the reach')
                self.enter(t, 'grasp')
            elif self.elapsed(t) > 12.0:
                self.report(f"[{t:7.2f}s] flight   no grasp fit after 12 s on station; "
                            f"flying home empty")
                self.enter(t, 'home')

        elif self.phase == 'grasp':
            # Frozen setpoint. The arm is moving and the COM-to-pelvis correction in
            # run_isaac is what keeps this from dragging the body around.
            self.station = self.frozen
            if mission.state in ('hold', 'done') or mission.state == 'failed' \
                    or self.elapsed(t) > GRASP_LIMIT:
                self.set_gains(type(self.controller).TRANSIT_GAINS
                               if self.controller is not None else (4., 3.3), 'back to transit')
            if mission.state in ('hold', 'done'):
                self.enter(t, 'home')
            elif mission.state == 'failed':
                self.report(f"[{t:7.2f}s] flight   grasp failed; flying home empty")
                self.enter(t, 'home')
            elif self.elapsed(t) > GRASP_LIMIT:
                # Hovering is not free: burning the rest of the sortie on an arm that
                # is stuck is how the robot ends up out of margin and on the floor.
                self.report(f"[{t:7.2f}s] flight   arm still in '{mission.state}' after "
                            f"{GRASP_LIMIT:.0f} s on a frozen hover; flying home")
                self.enter(t, 'home')

        elif self.phase == 'home':
            self.station = self.origin + np.array([0, 0, SCAN_RISE])
            if self.on_station(frames, self.station, velocity, tolerance=RETURN_XY):
                self.enter(t, 'gear')

        elif self.phase == 'gear':
            # Hold altitude while the legs come down. Descending onto half-extended
            # legs is a fall, and the joint drives are rate limited.
            self.station = self.origin + np.array([0, 0, SCAN_RISE])
            self.set_gear('down')
            if self.elapsed(t) >= GEAR_TIME:
                self.enter(t, 'land')

        elif self.phase == 'land':
            self.station = self.origin.copy()
            if pelvis[2] <= self.origin[2] + LAND_CLEAR:
                self.cut = True
                self.report(f"[{t:7.2f}s] flight   touchdown, jets off "
                            f"({float(np.linalg.norm(pelvis[:2]-self.origin[:2]))*1000:.0f} mm "
                            f"from where it lifted off)")
                self.enter(t, 'landed')

        else:                                   # landed
            self.station = self.origin.copy()
            self.cut = True

        return self.slew(self.station, dt)

    # ------------------------------------------------------------------- report
    def summary(self, frames=None):
        pelvis = None if frames is None else np.round(frames['pelvis'][0], 4).tolist()
        return dict(phase=self.phase, cut=bool(self.cut),
                    gear=None if self.walker is None else self.walker.gear,
                    origin=None if self.origin is None else np.round(self.origin, 4).tolist(),
                    grasp_station=None if self.grasp_station is None
                    else np.round(self.grasp_station, 4).tolist(),
                    frozen_station=None if self.frozen is None else np.round(self.frozen, 4).tolist(),
                    final_pelvis=pelvis,
                    return_error_mm=None if (self.origin is None or frames is None) else
                    round(float(np.linalg.norm(frames['pelvis'][0][:2] - self.origin[:2])) * 1000, 1),
                    phases=self.events)


def com_setpoint(desired_pelvis, frames):
    """Convert a pelvis setpoint into the COM setpoint the allocator regulates."""
    return np.asarray(desired_pelvis, dtype=float) + (center_of_mass(frames) - frames['pelvis'][0])
