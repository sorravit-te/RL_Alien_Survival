"""Render the Alien's poses from eight sides. Runs inside Blender, not the project's Python.

Usage (paths from tools.bake_alien_sprites):
    blender --background --factory-startup --python tools/render_alien_frames.py --
        assets/alien/source/Poses assets/alien/source/Colors/Green assets/alien/frames

View 0 looks at the Alien's front; each next view moves 45 degrees toward its right side,
matching renderers/alien_sprites.py. The floor sits on the bottom edge of every picture.
"""

import json
import sys
from math import pi
from pathlib import Path

import bpy
from mathutils import Vector

VIEWS = 8
SIZE = 320
WORLD_SIZE = 2.2  # metres across each picture; one metre is one ship cell
POSES = {"idle": ("Idle.fbx", 6), "walk": ("Walk.fbx", 12), "run": ("Run.fbx", 10)}  # file, frames
CAMERA_DISTANCE = 6.0
ALBEDO = "test_StingrayPBS1SG_AlbedoTransparency.png"
NORMAL = "test_StingrayPBS1SG_Normal.png"
SAMPLES = 32


def bare_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.render.resolution_x = scene.render.resolution_y = SIZE
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("ambient")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.5, 0.52, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.45
    scene.world = world
    return scene


def dress(mesh, skin):
    material = bpy.data.materials.new("alien_skin")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes["Principled BSDF"]
    color = nodes.new("ShaderNodeTexImage")
    color.image = bpy.data.images.load(str(skin / ALBEDO))
    bump = nodes.new("ShaderNodeTexImage")
    bump.image = bpy.data.images.load(str(skin / NORMAL))
    bump.image.colorspace_settings.name = "Non-Color"
    normal = nodes.new("ShaderNodeNormalMap")
    links.new(color.outputs["Color"], shader.inputs["Base Color"])
    links.new(bump.outputs["Color"], normal.inputs["Color"])
    links.new(normal.outputs["Normal"], shader.inputs["Normal"])
    shader.inputs["Roughness"].default_value = 0.42
    mesh.data.materials.clear()
    mesh.data.materials.append(material)


def camera_rig(scene):
    """A camera with its lights on one arm, so every view of the Alien is lit the same way."""
    arm = bpy.data.objects.new("arm", None)
    scene.collection.objects.link(arm)
    data = bpy.data.cameras.new("camera")
    data.type = "ORTHO"
    data.ortho_scale = WORLD_SIZE
    camera = bpy.data.objects.new("camera", data)
    scene.collection.objects.link(camera)
    camera.parent = arm
    camera.location = (0.0, -CAMERA_DISTANCE, WORLD_SIZE / 2)
    camera.rotation_euler = (pi / 2, 0.0, 0.0)
    scene.camera = camera
    for name, power, place in (("key", 2.6, (-1.5, -2.0, 3.0)), ("rim", 1.5, (2.0, 2.5, 2.0))):
        sun = bpy.data.lights.new(name, "SUN")
        sun.energy = power
        light = bpy.data.objects.new(name, sun)
        scene.collection.objects.link(light)
        light.parent = arm
        light.location = place
        light.rotation_euler = (Vector(place) * -1).to_track_quat("-Z", "Y").to_euler()
    return arm


def poses_of_cycle(count):
    """Evenly spaced moments of the longest action; the last frame repeats the first."""
    action = max(bpy.data.actions, key=lambda a: len(a.fcurves))
    first, last = action.frame_range
    return [round(first + (last - first) * index / count) for index in range(count)]


def render_pose(pose, file_name, count, model, skin, out):
    scene = bare_scene()
    bpy.ops.import_scene.fbx(filepath=str(model / file_name))
    dress(next(o for o in scene.objects if o.type == "MESH"), skin)
    arm = camera_rig(scene)
    for index, frame in enumerate(poses_of_cycle(count)):
        scene.frame_set(frame)
        for view in range(VIEWS):
            arm.rotation_euler = (0.0, 0.0, -view * 2 * pi / VIEWS)
            scene.render.filepath = str(out / f"{pose}_{index}_{view}.png")
            bpy.ops.render.render(write_still=True)


def main():
    extra = sys.argv[sys.argv.index("--") + 1 :]
    model, skin, out = (Path(argument).resolve() for argument in extra[:3])
    out.mkdir(parents=True, exist_ok=True)
    for pose, (file_name, count) in POSES.items():
        render_pose(pose, file_name, count, model, skin, out)
    sheet = {"world_size": WORLD_SIZE, "views": VIEWS, "frames": {p: c for p, (_, c) in POSES.items()}}
    (out / "frames.json").write_text(json.dumps(sheet), encoding="utf-8")
    print("RENDER DONE")


main()
