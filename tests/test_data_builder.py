import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from unibrain_agent.data import FIELDS, OriginalDataset, templates
from unibrain_agent.data_builder import build_dataset, read_manifest, eligible_subjects, resample_bundle, image_info, audit_dataset
from unibrain_agent.data_contract import model_contract


def fixture(root):
    grid = np.indices((8, 8, 8)).sum(axis=0).astype('f4')
    brain = np.zeros((8, 8, 8), dtype='f4');brain[1:7, 1:7, 1:7] = 1
    atlas = np.zeros_like(brain);atlas[brain > 0] = np.resize(np.arange(1, 117), int(brain.sum()))
    tissue = (atlas % 3 + 1) * brain
    def save(name, values, affine=None):
        path = root / (name + '.nii.gz')
        image = nib.Nifti1Image(values, np.diag([2., 2., 2., 1.]) if affine is None else affine)
        image.header.set_xyzt_units('mm');nib.save(image, path)
        return str(path)
    shared = {key: save(key, value) for key, value in [('brain', brain), ('tissue', tissue), ('atlas', atlas)]}
    image = save('template', grid)
    subjects = [{'id': 'subject' + str(i), 'label': i % 2, 'split': split, 'modality': 'T1w', 'supervision': 'manual',
                 'image': save(split, grid + i * .2), **shared} for i, split in enumerate(('train', 'val', 'test'))]
    return {'version': 1, 'intensity': 'preserve', 'labels': {'diagnosis': {'0': 'Control', '1': 'Case'},
            'tissue': {'0': 'Background', '1': 'Tissue A', '2': 'Tissue B', '3': 'Tissue C'}, 'atlas': 'AAL116'},
            'templates': {'image': image, **shared}, 'subjects': subjects}


class DataBuilderTests(unittest.TestCase):
    def test_contract_tracks_the_training_loader(self):
        self.assertEqual(model_contract()['fields'], list(FIELDS))
        self.assertEqual(model_contract()['shape'], [96, 96, 96])

    def test_nifti_build_matches_original_loader_and_records_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);manifest = fixture(root);target = root / 'built'
            result = build_dataset(manifest, target)
            self.assertTrue(result['passed'])
            row = OriginalDataset(target / 'adhd_train_sample.npy')[0]
            self.assertEqual(row[2].shape, (96, 96, 96))
            self.assertEqual(len(templates(target)['atlas_label_values']), 117)
            evidence = json.loads((target / 'geometry.json').read_text())
            self.assertEqual(len(evidence['subjects']), 3)
            self.assertGreater(evidence['subjects'][0]['qc']['atlas_inside_brain'], .99)
            self.assertEqual(evidence['subjects'][0]['source_files']['atlas']['interpolation'], 'nearest')
            changed = np.load(target / 'adhd_val_sample.npy', mmap_mode='r+')
            changed['raw_img_np'][:] = row[2];changed.flush();del changed
            with self.assertRaisesRegex(ValueError, 'Identical image'):
                audit_dataset(target)

    def test_missing_targets_and_wrong_modality_are_excluded_without_guessing(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = fixture(Path(folder))
            manifest['subjects'][0].pop('label')
            manifest['subjects'][1]['modality'] = 'bold'
            accepted, rejected = eligible_subjects(manifest)
            self.assertEqual(len(accepted), 1);self.assertEqual(len(rejected), 2)
            with self.assertRaisesRegex(ValueError, 'Every split'):
                build_dataset(manifest, Path(folder) / 'blocked')

    def test_subject_leakage_and_undefined_ontology_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);manifest = fixture(root)
            manifest['subjects'][1]['id'] = manifest['subjects'][0]['id']
            with self.assertRaisesRegex(ValueError, 'Repeated subject'):
                eligible_subjects(manifest)
            manifest['labels']['tissue'] = {}
            path = root / 'manifest.json';path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Declare diagnosis'):
                read_manifest(path)

    def test_world_coordinates_align_masks_instead_of_resizing_independently(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=fixture(root);row=copy.deepcopy(manifest['subjects'][0])
            mask=np.zeros((8,8,8),dtype='f4');mask[1:4,1:7,1:7]=1
            affine=np.diag([2.,2.,2.,1.]);affine[0,3]=4
            image=nib.Nifti1Image(mask,affine);image.header.set_xyzt_units('mm')
            path=root/'shifted.nii.gz';nib.save(image,path)
            row={'image':row['image'],'brain':str(path)}
            arrays,info=resample_bundle(row,root/'work')
            centre=np.argwhere(arrays['brain']>0).mean(axis=0)[0]
            self.assertGreater(centre,48)
            self.assertTrue(np.isin(arrays['brain'],[0,1]).all())
            self.assertEqual(info['source_files']['brain']['affine'][0][3],4)

    def test_unknown_units_nonfinite_data_and_label_interpolation_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=fixture(root);row=manifest['subjects'][0]
            image=nib.load(row['image']);image.header.set_xyzt_units('unknown');nib.save(image,row['image'])
            with self.assertRaisesRegex(ValueError,'millimetre'):image_info(row['image'])
            image.header.set_xyzt_units('mm');nib.save(image,row['image'])
            labels=np.ones((8,8,8),dtype='f4')*.5
            image=nib.Nifti1Image(labels,np.diag([2.,2.,2.,1.]));image.header.set_xyzt_units('mm');nib.save(image,row['brain'])
            with self.assertRaisesRegex(ValueError,'integer labels'):resample_bundle(row,root/'work')

    def test_dicom_conversion_is_explicit_and_requires_the_converter(self):
        from unibrain_agent.data_builder import convert_dicom
        with patch('unibrain_agent.data_builder.shutil.which',return_value=None):
            with self.assertRaisesRegex(ValueError,'Install dcm2niix'):convert_dicom('series',Path('unused'))

    def test_agent_receives_attached_audit_but_no_subject_records_or_paths(self):
        from unibrain_agent.data_workspace import DataWorkspace
        from unibrain_agent.catalog import DEFAULTS
        with tempfile.TemporaryDirectory() as folder:
            workspace=DataWorkspace(folder)
            workspace.selected={'path':'/private/subject-secret','audit':{'passed':True,'splits':{'train':{'count':3}},'files':{'private-path':'hash'}}}
            record={'id':'20261110_120000_abcdef12','mode':'discuss','sources':['/private/subject-secret'],'manifest':'','message':'Explain the attached data','selection':DEFAULTS,'previous':None,'events':[],'calls':[]}
            workspace.path(record['id']).parent.mkdir()
            inventory={'items':[],'counts':{'review':7},'dcm2niix_available':False}
            with patch('unibrain_agent.data_workspace.inspect_sources',return_value=inventory),patch('unibrain_agent.agents.Agent') as factory:
                factory.return_value.calls=[]
                factory.return_value.decide.return_value={'answer':'Checked','plan':[],'blockers':[],'model_changes':'None'}
                workspace.execute(record)
                context=factory.return_value.decide.call_args.args[1]
            self.assertTrue(context['attached_dataset_audit']['passed'])
            self.assertNotIn('subject-secret',json.dumps(context))
            self.assertNotIn('private-path',json.dumps(context))


if __name__ == '__main__':
    unittest.main()
