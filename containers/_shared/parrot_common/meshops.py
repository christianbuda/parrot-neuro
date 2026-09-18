"""Trimesh surface primitives shared across images."""
import numpy as np
import trimesh


def signed_clearance(reference, points):
    """AUTHORITATIVE signed clearance from `reference`: + outside, - inside.

    The SIGN must come from trimesh's signed_distance, whose pseudonormal test stays correct
    near edges and folds. A naive dot(point - closest, face_normal) does NOT: on a deformed
    shell it reads a deeply-inside vertex that happens to be closest to a nearby fold as
    "outside", so a repair scan never flags it and silently leaves an intersection.
    closest_point (see outward_dir) is only ever used for the push direction.

    Units follow the input meshes.
    """
    return -trimesh.proximity.signed_distance(reference, points)


def outward_dir(reference, points):
    """Outward push direction: normal of the closest `reference` triangle.

    This is the signed-distance gradient, from one cheap closest_point query, and needs no
    vertex correspondence between the two surfaces -- which is what makes an inflation loop
    converge in a few iterations even in a deep dent, and what lets it run on shells whose
    correspondence decimation has destroyed.
    """
    _, _, tri = trimesh.proximity.closest_point(reference, points)
    return reference.face_normals[tri]


def nesting_margins(inner, outer):
    """Both directions of the containment test between two shells.

    Returns (inner_inside, outer_outside): `inner_inside[i]` is how far inner.vertices[i] sits
    inside `outer`, `outer_outside[j]` how far outer.vertices[j] sits outside `inner`. Strict
    nesting means both are positive everywhere.

    Testing only one direction is not containment: `inner` can bulge through a large `outer`
    triangle while every `outer` vertex stays clear, which is how a badly deformed shell passes
    a one-sided repair and is then rejected by om_assemble.
    """
    return (-signed_clearance(outer, inner.vertices),
            signed_clearance(inner, outer.vertices))
