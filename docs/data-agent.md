# Data build agent

The main workspace contains a data agent, source inspector, converter and dataset attachment control. The agent reads the input contract, selected stage implementations, aggregate inventory and declared target availability. It can discuss missing supervision, source selection, split design and changes that the model or losses would need. Its recommendations are advisory; executable conversion and acceptance checks remain deterministic.

## Workflow

1. Connect your own backend and explicitly connect your AI account in AI settings. Inspection and conversion also work without an AI account.
2. Expand **Sources & conversion**. Enter folders on that backend, one per line. An empty source list inspects the configured UniBrain examples.
3. Click **Inspect sources**. Existing UniBrain datasets receive format and split checks. NIfTI headers are checked for dimensions, spatial units and affine validity. T1w filenames identify candidates only. DICOM folders require explicit series selection.
4. Ask the data agent a question. It receives the selected methods and aggregate evidence, not images, paths or individual subject records. Text you type is sent as your question.
5. For raw volumes, create the manifest below on the backend and enter its path. **Build dataset** creates an isolated dataset directory and records progress. The input files remain unchanged.
6. Review the result and click **Use dataset**. Prepare the pipeline again. The build pins this dataset's location and file hashes; the executor rejects changes after validation. Attaching another dataset invalidates the current UI build.

## Input contract

The current supervised trainer requires 3D T1 weighted images, binary diagnosis targets, brain masks, tissue labels 0 through 3, and AAL116 annotations. Image, tissue and atlas templates are required. All three subject splits must be nonempty. IDs must consistently identify a person across sources, including repeated visits. The application checks supplied identities and identical image content across splits; it cannot discover the same person under unrelated source identifiers.

Missing diagnoses or annotations are not synthesized. The agent can recommend annotation tools or a model/loss change. Generated annotations must be declared as pseudo labels, and their quality requires a separate assessment. A raw image by itself cannot satisfy this supervised training interface.

## Conversion manifest

Paths can be absolute or relative to the manifest. Supply your actual diagnosis and tissue class meanings, using the same ontology for every subject and template. Add train, val and test entries; the abbreviated example below contains one subject only.

```json
{
  "version": 1,
  "intensity": "preserve",
  "labels": {
    "diagnosis": {"0": "Your control definition", "1": "Your case definition"},
    "tissue": {"0": "Background", "1": "Your class 1", "2": "Your class 2", "3": "Your class 3"},
    "atlas": "AAL116"
  },
  "templates": {
    "image": "templates/T1w.nii.gz",
    "tissue": "templates/tissue.nii.gz",
    "atlas": "templates/aal116.nii.gz"
  },
  "subjects": [
    {
      "id": "subject001",
      "modality": "T1w",
      "split": "train",
      "label": 0,
      "supervision": "manual",
      "image": "subject001/T1w.nii.gz",
      "brain": "subject001/brain.nii.gz",
      "tissue": "subject001/tissue.nii.gz",
      "atlas": "subject001/aal116.nii.gz"
    }
  ]
}
```

`supervision` accepts `manual`, `pseudo` or `mixed`. `include: false` explicitly excludes a subject. Missing targets, incorrect modality and missing split assignments are recorded as exclusions. Duplicate subject IDs block the build. The user must resolve repeated acquisitions rather than allowing the application to choose a visit silently.

For DICOM, replace a subject's `image` with `dicom`, pointing to one selected T1 series directory. Install the official [dcm2niix converter](https://github.com/rordenlab/dcm2niix) on the backend. The conversion accepts exactly one NIfTI output; ambiguous multiseries outputs fail. Converter output remains in the private data job folder. DICOM conversion does not constitute a deidentification certification.

## Geometry and outputs

The executor uses [NiBabel physical resampling](https://nipy.org/nibabel/reference/nibabel.processing.html#nibabel.processing.resample_from_to). It canonicalizes image axes, preserves the reference image's physical field of view and resamples to 96 cubed. It samples annotations into that same physical grid with nearest neighbor interpolation. Each subject and the template retain their own recorded target affine. This operation does not estimate anatomical registration; that remains the pipeline registration stage. Anatomical quality, cross site consistency and semantic label correctness still require review.

The default intensity policy preserves intensity values. `brain_zscore` uses each image's own supplied brain mask and requires a template brain mask too. It estimates no statistics across splits. A chosen intensity policy changes the input distribution and should be kept consistent with training and evaluation.

Outputs are the original structured NPY split files and three template arrays. Private sidecars preserve the manifest, excluded subjects, source hashes, source and target affines, interpolation rules, brain bounds, annotation overlap measurements and validation results. Finite values, label ranges, AAL vocabulary, empty annotations, gross mask overlap, split identity and identical image content are checked. These checks establish loader compatibility, not anatomical correctness or predictive accuracy.

Jobs live under ignored `data_jobs`, configurable with `UNIBRAIN_DATA_JOBS`. Large inputs are processed one subject at a time into memory mapped output arrays. The current implementation still materializes each individual input volume. Dataset scans and content hashing can take time for large cohorts; this implementation has not been benchmarked at cohort scale.

Verification uses synthetic NIfTI volumes, shifted annotation affines, malformed targets and the original public NPY samples. Synthetic tests do not replace validation on a real external cohort. The DICOM command is supported but requires a separately verified local converter and series.
