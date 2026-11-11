# README images

Scripts that generate the images in [docs/img](../docs/img):

| Script | Output |
|---|---|
| `diagram.py` | `gpushift-light.svg`, `gpushift-dark.svg`: the overview diagram at the top of the README (GitHub shows the one matching the reader's theme). No dependencies. |
| `clip.py` | `gpushift-hook.gif`: an 8-second clip of the hook stopping a direct launch. The text is gpushift's real hook message and submit output. Needs Pillow. |

```bash
pip install pillow
python demos/diagram.py docs/img
python demos/clip.py docs/img/gpushift-hook.gif
```
