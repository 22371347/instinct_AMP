"""渲染**默认姿态**（MJCF keyframe `home`）的多视角参考图。

用途：把这张图和训练/回放里模型的"静止姿态"并排对比 ——
`track_default_pose_on_stand` 就是往这个姿态拉，AMP 参考动作的起始帧也是它。

只做**离屏渲染**（不启动交互式模拟器、不做任何控制测试）：

    MUJOCO_GL=egl python instinct/tools/render_default_pose.py
    # 若 EGL 不可用：MUJOCO_GL=osmesa python instinct/tools/render_default_pose.py

输出：`instinct/assets/g1_default_pose.png`（front / side / 3-4 三视角横向拼接）
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
DEFAULT_XML = os.path.join(ASSETS, "g1_29dof_popsicle_play.xml")
DEFAULT_OUT = os.path.join(ASSETS, "g1_default_pose.png")

#: (名字, 方位角, 仰角) —— 方位角 180 = 正面（朝 +x 看回去）
VIEWS: tuple[tuple[str, float, float], ...] = (
    ("front", 180.0, -8.0),
    ("side", 90.0, -8.0),
    ("three_quarter", 135.0, -12.0),
)


def render(
    xml: str, out: str, key: str, width: int, height: int, distance: float, lookat_z: float
) -> None:
    import mujoco  # 延后导入：MUJOCO_GL 必须在 import 前设好

    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)

    kid = -1
    if model.nkey > 0:
        kid = 0
        for i in range(model.nkey):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
            if name == key:
                kid = i
                break
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
        print(f"[pose] 使用 keyframe #{kid} name={mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, kid)!r}")
    else:
        mujoco.mj_resetData(model, data)
        print(f"[pose] ⚠️ 没找到 keyframe {key!r}，回退到模型自带的 qpos0")
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    #: 看骨盆高度附近（root z = 0.82）
    cam.lookat[:] = [0.0, 0.0, lookat_z]
    cam.distance = distance

    panels = []
    for name, az, el in VIEWS:
        cam.azimuth, cam.elevation = az, el
        renderer.update_scene(data, camera=cam)
        panels.append(renderer.render().copy())
        print(f"  ✅ {name:14s} azimuth={az:6.1f} elevation={el:5.1f}")
    renderer.close()

    img = np.concatenate(panels, axis=1)
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(img).save(out)
    except Exception:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.image as mpimg

        mpimg.imsave(out, img)
    print(f"[out ] {out}  ({img.shape[1]}x{img.shape[0]}, {os.path.getsize(out) / 1024:.1f} KB)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="渲染默认姿态多视角参考图（离屏）")
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--key", default="home", help="keyframe 名字（默认 home = popsicle 默认站姿）")
    #: ⚠️ MuJoCo 离屏 framebuffer 默认只有 640x480，超过会直接报错
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--distance", type=float, default=2.6)
    ap.add_argument("--lookat-z", type=float, default=0.80, help="相机注视点高度（骨盆在 0.82）")
    args = ap.parse_args(argv)

    print(f"[gl  ] MUJOCO_GL={os.environ.get('MUJOCO_GL', '(未设置)')}")
    render(args.xml, args.out, args.key, args.width, args.height, args.distance, args.lookat_z)
    return 0


if __name__ == "__main__":
    sys.exit(main())
