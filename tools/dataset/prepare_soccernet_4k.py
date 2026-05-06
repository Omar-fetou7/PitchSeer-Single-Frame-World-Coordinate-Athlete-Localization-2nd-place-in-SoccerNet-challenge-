#!/usr/bin/env python
"""Prepare 4K SoccerNet annotations for ViTPose COCO format."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert 4K SoccerNet annotations to ViTPose COCO format.'
    )
    parser.add_argument(
        '--ann-dir',
        required=True,
        help='Path to the already-extracted annotations directory (contains train.json, val.json, etc.).')
    parser.add_argument(
        '--output-root',
        default='data/soccernet_4k',
        help='Output directory for converted annotations.')
    return parser.parse_args()


def flatten_keypoints(points):
    flat = []
    for point in points:
        flat.extend(point)
    return flat


def convert_annotations(src_data):
    converted_images = []
    for image in src_data['images']:
        converted_image = {
            'id': image['id'],
            'file_name': image['file_name'],
            'width': image['width'],
            'height': image['height'],
        }
        for optional_key in ('camera_matrix', 'undist_poly', 'dist_poly'):
            if optional_key in image:
                converted_image[optional_key] = image[optional_key]
        converted_images.append(converted_image)

    converted_annotations = []
    for ann in src_data['annotations']:
        keypoints = ann['keypoints']
        flat_keypoints = flatten_keypoints(keypoints)
        bbox = ann['bbox']
        converted_annotations.append({
            'id': ann['id'],
            'image_id': ann['image_id'],
            'category_id': ann['category_id'],
            'bbox': bbox,
            'area': float(ann['area']),
            'iscrowd': 0,
            'num_keypoints': sum(point[2] > 0 for point in keypoints),
            'keypoints': flat_keypoints,
            'keypoints_3d': ann.get('keypoints_3d'),
            'position_on_pitch': ann.get('position_on_pitch'),
        })

    categories = [{
        'id': 1,
        'name': 'person',
        'supercategory': 'person',
        'keypoints': ['body_anchor', 'ground_contact'],
        'skeleton': [[1, 2]],
    }]

    return {
        'info': {
            'description': 'SoccerNet 4K annotations in ViTPose COCO format',
            'version': '1.0',
        },
        'licenses': [],
        'images': converted_images,
        'annotations': converted_annotations,
        'categories': categories,
    }


def convert_split(ann_dir, split, out_json):
    src_json = ann_dir / f'{split}.json'
    if not src_json.is_file():
        raise FileNotFoundError(f'Missing annotation file: {src_json}')

    with open(src_json) as f:
        src_data = json.load(f)

    converted = convert_annotations(src_data)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(converted))
    print(f'Wrote {out_json}')


def main():
    args = parse_args()
    ann_dir = Path(args.ann_dir)
    output_root = Path(args.output_root)
    ann_out_root = output_root / 'annotations'

    for split in ('train', 'val', 'test'):
        src_json = ann_dir / f'{split}.json'
        if src_json.is_file():
            convert_split(ann_dir, split, ann_out_root / f'{split}.json')
        else:
            print(f'Skipping {split} (not found: {src_json})')


if __name__ == '__main__':
    main()
