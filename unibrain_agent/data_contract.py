def model_contract(selection=None):
    from .catalog import DEFAULTS, catalog
    selected = selection or DEFAULTS
    methods = [{"stage": stage["name"], **{key: value for key, value in next(option for option in stage["options"] if option["id"] == selected[stage["id"]]).items() if key in {"id", "label", "adaptation"}}} for stage in catalog()["stages"]]
    return {"modality": "3D T1 weighted structural MRI", "shape": [96, 96, 96], "fields": ["id", "label", "raw_img_np", "raw_img_brain_mask_np", "raw_img_gm_mask_np", "raw_img_aal_mask_np"],
            "splits": ["train", "val", "test"], "diagnosis": "User supplied binary target: 0 or 1", "stage_requirements": methods,
            "supervision": {"brain": "Binary brain mask", "tissue": "Integer tissue labels 0 through 3",
                            "atlas": "AAL116 labels using exactly the template vocabulary"},
            "templates": {"image": "template_img_orig_96.npy", "tissue": "template_img_gm_mask_orig_96.npy", "atlas": "template_img_aal_mask_orig_96.npy"}, "selection": selection or DEFAULTS,
            "constraints": ["Subject identity must be consistent across sources and splits.",
                            "Template atlas must contain background and all 116 regions.",
                            "Physical resampling is not anatomical registration.",
                            "Never infer diagnosis labels from image appearance or filenames.",
                            "Missing supervision blocks the current supervised trainer.",
                            "Generated or propagated masks must be explicitly identified as pseudo labels."],
            "evidence": ["unibrain_agent/data.py:FIELDS,load_split,templates",
                         "unibrain_agent/pipeline.py:loss_terms", "pinned UniBrain:model.py,train.py"]}
