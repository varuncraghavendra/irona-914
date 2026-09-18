# Joint design and actuator assumptions

## Articulation

The 62 revolute joints form a single free-floating tree with 63 massive links. Twelve joints articulate the legs, four articulate the waist and neck, fourteen articulate the two arms, and thirty-two articulate the two hands. The previous prismatic grippers have been removed.

Each arm has offset shoulder pitch, roll and yaw pivots; elbow flexion; forearm pronation/supination; wrist pitch; and wrist roll. The axes are separated in space rather than all sharing a single spherical center. CAD horns, shafts and races follow their actual parent or child link. Fingers have separate rigid bodies, box contact proxies and explicit joint stops. Four fingers have MCP/PIP/DIP flexion, while the thumb has opposition, basal spread, MCP and IP flexion.

The masses and COMs form a 15.086 kg engineering budget. Diagonal inertias are box-equivalent calculations about each specified COM, not exact mass properties of a finished motorized assembly. Collision boxes are smaller/simpler than the exterior. Nearby links are filtered to avoid coincident carrier contact. Full-range CAD interference has not been certified.

## DYNAMIXEL basis

| Application, per arm | Reference | Manufacturer reference values | Simulation choice |
|---|---|---|---|
| Shoulder pitch/roll/yaw | XM540-W270 | 10.6 Nm stall and 30 rpm no-load at 12 V; 165 g; 33.5 × 58.5 × 44 mm | 4.0 / 4.0 / 3.5 Nm caps; 2 rad/s |
| Elbow, forearm, two wrist axes | XM430-W350 | 4.1 Nm stall and 46 rpm no-load at 12 V; 82 g; 28.5 × 46.5 × 34 mm | 1.5 / 1.0 / 0.85 / 0.65 Nm caps; 3 rad/s |
| Conceptual remote finger actuation | XL330-M288 | 0.52 Nm stall, 103 rpm no-load at 5 V | Distributed 0.025–0.10 Nm virtual joint caps; 2–2.5 rad/s |

References: ROBOTIS [XM540-W270](https://emanual.robotis.com/docs/en/dxl/x/xm540-w270/), [XM430-W350](https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/) and [XL330-M288](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/).

Stall torque is a momentary limit, not continuous operating torque. The project caps are conservative **assumptions**, not manufacturer-certified continuous ratings. A hardware build needs load, thermal, duty-cycle, transmission and voltage sizing. The legs and torso retain custom high-torque drive assumptions; they are not claimed to be direct-drive XM430 limbs.

The hand has 16 simulated joint drives. Software poses coordinate them like a tendon-driven hand, but the simulator does not contain cable routing, pulleys, tendon elasticity, a six-motor hardware transmission or a manufactured ROBOTIS five-finger product. Independent joint-state commands are accepted. Do not infer that an XL330 fits inside each phalanx.

## Physics and controls

- USD defines positive mass/inertia, bounded angular travel, force-limited stiffness/damping and maximum joint speed for every axis.
- `physxJoint:jointFriction` is a dimensionless PhysX joint-friction coefficient: 0.04 for arm joints, 0.02 for fingers and 0.03 for original joints. These are nominal values, not Coulomb friction torques or measured gearbox losses.
- Angular USD gains are converted from Nm/rad and Nm·s/rad to their per-degree USD equivalents. Joint limit values in USD are degrees; controllers and URDF use radians.
- The common target limiter clips both range and commanded angular rate. Native drive saturation limits simulated torque. Command-rate limiting is not a full electrical motor or torque-speed model.
- The hand demo uses smooth target synergies with open/closed dwell periods. Physics determines the actual link motion and contacts. Targets are not visual animation transforms.
- The root is initialized once and remains dynamic thereafter. Flight uses bounded external wrenches at the feet and torso.

Backlash, encoder quantization, bus latency, gearbox efficiency, electrical current, temperature derating, cable compliance and bearing preload are not simulated. The revised independent MuJoCo runs use the same kinematics, mass budget, joint limits and drive gains, with a different contact solver and no claim of PhysX joint-friction equivalence.

## Printing and hardware

The new shafts, horns, fastener details, palm pieces and phalanges are CAD/print prototypes. The finger pilot bores and visual knuckles demonstrate a proposed joint arrangement; they do not establish a qualified pinned linkage with retained metal hardware. Some small metal-colored parts should be replaced by actual purchased fasteners for a physical prototype. The STEP assembly and part manifest identify their intended locations.

For powered hardware, resolve motor packaging, complete bearing supports, tolerances, shaft retention, tendon transmissions, load paths and wiring before using the motion limits in hardware. The current deliverable supports simulation and a static full-size display.
