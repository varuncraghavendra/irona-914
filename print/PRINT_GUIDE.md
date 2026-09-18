# Irona 914 — printing and static assembly

## Scope and scale

Print at **100% scale, millimetres**. The fully assembled neutral exterior is
914.4 mm high. STL files contain only geometry and cannot enforce a unit system;
select mm in the slicer if prompted. USD uses metres and should not be sent
directly to a slicer.

This kit is suitable for a glued, static display and for evaluating cosmetic
surfaces around a future robot chassis. Its joints are visual covers and its
nozzles are decorative. It does not contain actuator mounts, working bearings,
qualified working five-finger linkages or operating jet hardware.

## Before printing the full kit

1. Open `cad/irona_exterior.step` in a STEP-capable CAD viewer. The model is already
   assembled, in mm, with named components.
2. Read `parts.csv` to identify the numbered shell files and desired colours.
   Every exported part has been checked for closed mesh geometry and an envelope
   no larger than 200 mm per axis. Check the slicer's brim and support footprint.
3. Start with one shin half and a joint cover. Check wall quality, dimensional
   accuracy and the appearance of the seam before committing to all parts.
4. Dry-fit a pair of shell halves and a core sleeve. The cosmetic seating cuts
   are nominal, with no printer-specific fit allowance. Sand or adjust CAD for
   your printer rather than forcing a tight assembly.

Suggested starting settings for a static indoor display: 0.4 mm nozzle, 0.2 mm
layers and PLA or PETG. The large cosmetic skins already have walls in the CAD;
use enough perimeters to fill those walls. The static core can use 4–6 perimeters
and moderate infill. These are slicing starting points, not strength ratings.
Orient broad split faces down where possible. Curved interiors, finger knuckles, sensor housings and
some joint covers may need supports. Individual parts have not been physically
printed or individually optimized for support-free manufacture.

## Assembly references

The frame is +X forward, +Y the robot's left, +Z up. Names such as `left_shin`
refer to the robot's left. The `parts.csv` translations reconstruct each STL's
position in the CAD assembly: add the three `assembly_t*_mm` values to its
vertices. No rotations are required. All STLs are translated to positive local
coordinates for slicing, so importing them all at the origin will not assemble
the robot automatically.

`assembly_transforms.json` contains the same exterior translations.
`display_armature/transforms.json` contains the separate core segment positions.
The two STEP files provide the full exterior and armature references.

## Static display build order

1. Glue the segmented static armature in the neutral pose shown in its STEP file.
   Align its spine, shoulder bar, pelvis bar, arms and legs against the CAD
   dimensions. Use flat temporary jigs so the feet remain parallel.
2. Reinforce accessible 16 mm core-rod butt joints with the provided sleeve.
   Its nominal bore is 16.4 mm. Sand it to fit and bond it in place. The smaller
   arm rods and crossbar junctions require adhesive and local reinforcement.
3. Assemble boots and soles around the foot supports. The soles have cosmetic
   nozzle openings. Use a rigid display base or a discreet rear support to keep
   the tall display from tipping.
4. Fit the shin and thigh skins around the core. Use small foam or printed shims
   between the inner shell and core where needed. Dry-fit before bonding.
5. Assemble pelvis and skirt panels. Work upward to the torso, shoulder covers,
   arm shells, bearing details and palms.
6. Fit neck covers, neutral head halves and the camera-bar mock-up. Assemble the
   lidar base and top cap using a transparent spacer through the open scan band.
   Keep the final cap top at 914.4 mm. The CAD intentionally omits an opaque
   window around the sensor origin. Printed plastic is not an optical sensor window.
7. Assemble each five-finger hand in its open display pose. Finger phalanges,
   pads, knuckles and pin/fastener details are individually listed in the manifest.
   Small knuckle bores are 2.4 mm pilots, not a proven retained hinge mechanism.
   Use a jig, adhesive and local support for a static display. For movable fingers,
   resolve a metal-pin linkage and clearance in CAD before printing the full hand.
8. Fit backpack shells and decorative nozzle rings last, then finish seams.

The included `seam_strap_print_as_needed.stl` is a small glue-on reinforcement
strip for the inside of straight cosmetic seams. Print as many as the assembly
needs. It is not a structural load connector. Trim strips at tight corners and
keep them clear of mating features.

## Finishing and a future powered build

The colour labels in the CSV correspond to the render: silver, bright silver,
graphite, ivory and copper. Paint small details after bonding if separate parts
are inconvenient. Mating seats remove most same-link geometric overlap; glue
lines, tolerances, access and small spacers remain a physical prototyping task.

A powered version needs a separately designed frame with selected actuators and
bearings. Do not drill arbitrary motor holes into these covers and treat them as
load-bearing mounts. Once the frame and electronics exist, replace the assumed
simulation masses and inertias with measured or CAD-derived assembly values.
No polymer temperature specification or structural design is provided for real
jet exhaust. Keep the flying function in simulation for this version.

## New arm and sensor parts

The part manifest identifies prototype shafts, bearing races, horns, fasteners,
phalanges and sensor housings by role. Small screw details can be omitted from
printing and represented with paint or purchased hardware. The D455 housing,
optical windows and lidar body are envelope mock-ups. Remove mock-ups when
engineering mounts for real sensors; no mounting-hole compatibility is certified.

Print the four larger finger segments and one palm half first to check the small
bore/edge quality. A finer nozzle or resin process may reproduce small details
better. The source is parametric so you can thicken or simplify them. The neutral
static armature has been moved outward to the new arm centerlines; it does not
support powered arm motion and must not be installed into a moving simulation.
