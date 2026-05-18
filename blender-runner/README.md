# blender-runner

Headless Blender runner for Ryvion render jobs.

The node mounts a work directory at `/work`. The runner reads `/work/job.json`,
renders a `.blend` scene that is already present in `/work`, writes rendered
frames under `/work/output`, and writes `/work/receipt.json` plus
`/work/metrics.json`.

Minimal job:

```json
{
  "scene_path": "/work/scene.blend",
  "frame": 1,
  "output_format": "PNG"
}
```

Supported fields:

- `scene_path`: absolute path under `/work`; default `/work/scene.blend`
- `frame`: single frame to render
- `frame_start` and `frame_end`: frame range
- `output_dir`: absolute path under `/work`; default `/work/output`
- `output_prefix`: output basename prefix; default `frame`
- `output_format`: `PNG`, `JPEG`, `OPEN_EXR`, or `TIFF`
- `engine`: optional Blender engine value, for example `CYCLES`
- `samples`: optional Cycles sample count
- `resolution_x` and `resolution_y`: optional render resolution
