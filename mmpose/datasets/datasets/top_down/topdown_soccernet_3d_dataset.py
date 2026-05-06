from .topdown_coco_dataset import TopDownCocoDataset
from ...builder import DATASETS


@DATASETS.register_module()
class TopDownSoccerNet3DDataset(TopDownCocoDataset):
    """TopDownCocoDataset extended with per-image camera parameters and
    per-annotation ground-plane position, required for the 3D projection loss.

    The three extra fields added to every db record:
        camera_matrix      (list[list[float]] | None): 3×4 or 4×4 matrix from
                           the image JSON.  The loss slices [:3] if 4×4.
        undist_poly        (list[float] | None): undistortion polynomial from
                           the image JSON (passed directly to sskit).
        position_on_pitch  (list[float, float] | None): [x, y] world coords of
                           the ground_contact point.

    Existing behaviour is completely unchanged when these fields are absent from
    the JSON (all three default to None) so the dataset can still be used with
    annotation files that lack camera metadata.
    """

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        img_ann = self.coco.loadImgs(img_id)[0]
        camera_matrix = img_ann.get('camera_matrix')
        undist_poly = img_ann.get('undist_poly')

        width = img_ann['width']
        height = img_ann['height']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # Replicate the parent's bbox-validity filter (same order guaranteed)
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                valid_objs.append(obj)

        # Replicate the parent's keypoint filter to get position_on_pitch in
        # the exact same order as the records the parent returns.
        pitch_positions = []
        for obj in valid_objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            pitch_positions.append(obj.get('position_on_pitch'))

        recs = super()._load_coco_keypoint_annotation_kernel(img_id)

        for i, rec in enumerate(recs):
            rec['camera_matrix'] = camera_matrix
            rec['undist_poly'] = undist_poly
            rec['position_on_pitch'] = (pitch_positions[i]
                                        if i < len(pitch_positions) else None)
            # TopDownAffineMosaicAug overwrites 'center' and 'scale' with crop
            # dimensions at the end of every forward pass. Save the original 4K
            # bounding-box values here so the 3D loss can use them for the UDP
            # inverse transform (heatmap → 4K pixel).
            rec['orig_center'] = rec['center'].copy()
            rec['orig_scale'] = rec['scale'].copy()
        return recs
