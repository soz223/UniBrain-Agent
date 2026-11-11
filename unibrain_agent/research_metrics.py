"""Region and surface metrics with explicit spatial units and empty-mask rules."""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def region_metrics(prediction, reference, spacing, units="mm", labels=None):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape or prediction.ndim != 3:
        raise ValueError("Masks must share the same three dimensional grid")
    if not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError("Masks must be finite")
    if not np.equal(prediction, np.rint(prediction)).all() or not np.equal(reference, np.rint(reference)).all():
        raise ValueError("Discrete label masks are required")
    if units not in ("mm", "voxel") or len(spacing) != 3 or any(not np.isfinite(x) or x <= 0 for x in spacing):
        raise ValueError("Explicit positive voxel spacing and mm or voxel units are required")
    if units == "voxel" and tuple(spacing) != (1, 1, 1):
        raise ValueError("Voxel units use unit spacing")
    labels = sorted(set(np.unique(reference)) | set(np.unique(prediction))) if labels is None else labels
    rows = []
    connectivity = generate_binary_structure(3, 1)
    for label in labels:
        if label == 0:
            continue
        p, r = prediction == label, reference == label
        count = int(p.sum() + r.sum())
        row = {"label": int(label), "predicted_voxels": int(p.sum()), "reference_voxels": int(r.sum()),
               "dice": 2 * int((p & r).sum()) / count if count else None, "hd95": None, "assd": None}
        if p.any() and r.any():
            ps = p & ~binary_erosion(p, connectivity, border_value=0)
            rs = r & ~binary_erosion(r, connectivity, border_value=0)
            a = distance_transform_edt(~rs, sampling=spacing)[ps]
            b = distance_transform_edt(~ps, sampling=spacing)[rs]
            row.update(status="evaluated", hd95=float(max(np.percentile(a, 95), np.percentile(b, 95))),
                       assd=float((a.sum() + b.sum()) / (len(a) + len(b))))
        else:
            row["status"] = "both_empty" if not count else "one_empty"
        rows.append(row)
    means = {key: float(np.mean([r[key] for r in rows if r[key] is not None]))
             if any(r[key] is not None for r in rows) else None for key in ("dice", "hd95", "assd")}
    return {"units": units, "spacing": list(spacing), "regions": rows, "mean": means,
            "surface_valid_regions": sum(r["status"] == "evaluated" for r in rows),
            "one_empty_regions": sum(r["status"] == "one_empty" for r in rows),
            "definition": "HD95=max of directional 95th percentiles; ASSD=pooled surface voxel mean; six-neighbour surfaces",
            "boundary": "Geometry must be aligned before scoring; empty surface distances are undefined and excluded with explicit counts"}
