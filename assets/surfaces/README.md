# Surface textures (for perception rendering, CC0)

Small textures of the workspace surface used by the simulated perception rendering (`cleanproof/sim/dirt.py`, `configs/sim.yaml` `dirt.surfaces`).

- Each covers exactly the 0.24 m × 0.30 m workspace at 480 × 600 pixels (0.5 mm / pixel). Rows run along base +x, columns along base +y.
- Cut from the source images below by `scripts/make_surface_textures.py`. The source images are not in the repo; download the originals from their upstream sources.

| File | Surface (force-table key) | Source image | Author | License | Source |
|---|---|---|---|---|---|
| `sealed_wood.png` | sealed_wood | `grey_oak_veneer_01_diff_4k.jpg` (whole image = 1.0 m) | Jenelle van Heerden | CC0 1.0 | https://polyhaven.com/a/grey_oak_veneer_01 |
| `rough_stone.png` | rough_stone | `Granite002A_4K-JPG_Color.jpg` (whole image = 0.4 m) | ambientCG | CC0 1.0 | https://ambientcg.com/view?id=Granite002A |
| `stainless_steel.png` | stainless_steel | `Metal011_4K-JPG_Color.jpg` (whole image = 1.0 m) | ambientCG | CC0 1.0 | https://ambientcg.com/view?id=Metal011 |

The whiteboard is a single flat color and has no texture. `surfaces.json` records the generation parameters and the mean color of each texture.
