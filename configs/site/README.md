# On-site site configuration (real robot)

When `CLEANPROOF_SITE=<site>` is set, the YAML files under `configs/site/<site>/` with the same name as those in `configs/` are deep-merged onto the default configuration
(`cleanproof/common/config.py` `load_yaml`). Simulation and tests do not set this variable, so their behavior is unchanged.

`scripts/calibrate_site.py` generates these on site:

| File | Content |
|---|---|
| `zones.yaml` | Measured countertop height, work-area corners and size, number of rows and columns, survey pose `survey_q`, site forbidden zones |
| `robot.yaml` | Measured polishing pad radius etc. (only keys that were measured) |
| `cameras.yaml` | Locked exposure / gain / white balance |
| `calib.json` | Countertop homography (base-frame xy → fixed camera pixels), corresponding points and reprojection error |
| `reference.png` | Top-down reference image of the clean countertop (baseline for residue detection) |
| `calib_overlay.jpg` | Grid overlaid on the raw frame, for manual checking |

Do not fill in values that were never measured; keep placeholder values in the default files under `configs/`.
