# Robot product photo sources (Wikimedia Commons)

The license and author of every image were checked against the Commons file description page (2026-10-28, via the Commons API and the description page metadata). File names are the files under `raw/`, downloaded unchanged.

| Model key | File | Title | Author | License (with version) | File description page | Attribution / share-alike required | Notes |
|---|---|---|---|---|---|---|---|
| spot_arm | raw/spot_arm_1.jpg | FEL CVUT Technicka 2 Praha 6 Interier R | Mojmir Churavy | CC0 1.0 | https://commons.wikimedia.org/wiki/File:FEL_CVUT_Technicka_2_Praha_6_Interier_R.jpg | No / No | Spot with arm (the description says arm + camera); indoor tiled floor, cluttered background, no people, whole robot in frame; portrait 2776x3424 |
| spot_arm | raw/spot_arm_2.jpg | Boston Dynamics Spot in Milan | Saggittarius A | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Boston_Dynamics_Spot_in_Milan.jpg | Yes / Yes | No arm; banner and fence in the exhibition background, no faces; whole robot; 2736x2791 |
| spot_arm | raw/spot_arm_3.jpg | Spot by Boston Dynamics | Jonte | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Spot_by_Boston_Dynamics.jpg | Yes / Yes | No arm; forest path background, no people; whole robot; 6000x4000 |
| ur5e | raw/ur5e_1.jpg | Robot Lengan Industri UR5e | Dzikra muhammad Imtiyaz | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Robot_Lengan_Industri_UR5e.jpg | Yes / Yes | Title says UR5e; arm mounted on a cart with drawers, cluttered indoor office background, no people; whole robot; 4032x3024 |
| panda | raw/panda_1.jpg | Franka Emika2 | Ims | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Franka_Emika2.jpg | Yes / Yes | Side view of a Franka Emika arm (description: Franka Emika arm of the company KBee); trade-show background, a person's torso and hands on the right of the frame, no clear face; whole robot; 6000x4000 |
| panda | raw/panda_2.jpg | Franka Emika3 | Ims | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Franka_Emika3.jpg | Yes / Yes | Front view, fairly cluttered trade-show background but no people; with gripper; portrait 4000x6000 |
| rizon4 | raw/rizon4_1.jpg | Flexiv RIZON | MichelleACipriano | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Flexiv_RIZON.jpg | Yes / Yes | Description only says Flexiv Rizon, not whether it is a Rizon 4; a person's suited torso and hands on the left (no face), trade-show background; whole robot but cropped; 3000x2000 |
| iiwa14 | raw/iiwa14_1.jpg | KUKA LBR iiwa | KUKA Laboratories GmbH | CC BY-SA 3.0 | https://commons.wikimedia.org/wiki/File:KUKA_LBR_iiwa.jpg | Yes / Yes | Official render/photo style image of the whole robot on a clean white gradient background, but only 398x234; the description does not distinguish iiwa 7 / 14 (looks like the LBR iiwa series, cannot confirm the 14 kg version) |
| sawyer | raw/sawyer_1.tif | Sawyer Portrait-Dark-Eyes-Master | RT Robotics | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Sawyer_Portrait-Dark-Eyes-Master.tif | Yes / Yes | Studio shot of the whole robot on a clean dark-grey gradient background, best suited for the card; original is a TIFF (199 MB locally, 7047x4716, confirmed to decode fully), must be converted to JPG/PNG and downscaled before use |
| sawyer | raw/sawyer_2.jpg | Schramberg - Sawyer 3 | Jeff Green/Rethink Robotics | CC BY 4.0 | https://commons.wikimedia.org/wiki/File:Schramberg_-_Sawyer_3.jpg | Yes / No | Factory line scene, cluttered background; whole robot; no faces; 6016x4016 |
| sawyer | raw/sawyer_3.jpg | Robot colaborativo Sawyer | Mizael Contreras | CC BY-SA 4.0 | https://commons.wikimedia.org/wiki/File:Robot_colaborativo_Sawyer.jpg | Yes / Yes | Shot at a trade show, people in the background (far away, no close-up faces); whole robot with base; portrait 2448x3264 |

## Not found (no clearly labeled openly licensed image of the model on Commons)

ur3e, ur10e, fr3 (Franka Research 3), kinova_gen3, xarm7, lite6.

Notes: Commons has `File:Brazo robótico Jaco KINOVA.jpg` (CC BY-SA 4.0, Mgomezrobotnik), which is a Kinova Jaco, not a Gen3, so it was not downloaded; it also has `File:UR16e robot arm.png` (CC BY-SA 4.0, Auledas), which is a UR16e, not one of the listed models, so it was not downloaded either.

## Images used on the software device page (`cutout/`)

Decision: show all 12 models; models with a licensed original are cut out onto a white background, the others use a white-background studio-style render (`assets/robots/<model>.png`). Cutouts were made with rembg (ISNet) and every one was inspected; only two reached product-image quality:

| Model key | Image used | Source file | Processing | License |
|---|---|---|---|---|
| spot_arm | `cutout/spot_arm.png` | raw/spot_arm_1.jpg | Background removed (mask thresholded to restore the arm forearm); the Boston Dynamics wordmark on the body and two university stickers are partly blurred (same approach as the SMRT photo in the Singapore section) | CC0 1.0, no attribution required (still listed on the About page) |
| sawyer | `cutout/sawyer.png` | raw/sawyer_1.tif | Background removed, downscaled to 900 px | CC BY-SA 4.0: credit RT Robotics, state "background removed", this derivative is released under the same license |

Why renders are used instead of photos: iiwa14_1 (official multi-pose ghosted image, 398 px), ur5e_1 (mounted on a cluttered cart), rizon4_1 and panda_1 (a person's hands in frame), panda_2 (dark background left between the joints after cutout).

The original `raw/` directory is not in git (it holds the 199 MB TIFF); re-download from the links in the table above.
