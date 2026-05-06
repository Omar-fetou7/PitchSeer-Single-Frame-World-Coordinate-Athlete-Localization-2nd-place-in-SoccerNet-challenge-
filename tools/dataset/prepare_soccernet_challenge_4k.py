#!/usr/bin/env python3
"""Prepare 4K challenge metadata for SoccerNet SpiideoSynLoc."""

import argparse
import json
from pathlib import Path
import zipfile


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--challenge-json',
        required=True,
        help='4K challenge metadata JSON with image ids and calibration.')
    parser.add_argument(
        '--images-root',
        required=True,
        help='Root directory containing challenge.zip.')
    parser.add_argument(
        '--output-json',
        required=True,
        help='Where to write the 4K metadata JSON.')
    parser.add_argument(
        '--skip-extract',
        action='store_true',
        help='Skip extracting challenge.zip if images are already unpacked.')
    return parser.parse_args()


def ensure_extracted(images_root: Path):
    challenge_dir = images_root / 'challenge'
    if challenge_dir.is_dir() and any(challenge_dir.glob('*.jpg')):
        print(f'Found extracted challenge images in {challenge_dir}')
        return challenge_dir

    zip_path = images_root / 'challenge.zip'
    if not zip_path.is_file():
        raise FileNotFoundError(f'Missing archive: {zip_path}')

    print(f'Extracting {zip_path} -> {images_root}')
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(images_root)
    return challenge_dir


def convert_images(src_images, target_width, target_height):
    converted_images = []
    for image in src_images:
        converted_image = {
            'id': image['id'],
            'file_name': image['file_name'],
            'width': target_width,
            'height': target_height,
        }
        for optional_key in ('camera_matrix', 'undist_poly', 'dist_poly'):
            if optional_key in image:
                converted_image[optional_key] = image[optional_key]
        converted_images.append(converted_image)
    return converted_images


def main():
    args = parse_args()
    challenge_json = Path(args.challenge_json)
    images_root = Path(args.images_root)
    output_json = Path(args.output_json)

    if not args.skip_extract:
        ensure_extracted(images_root)

    with challenge_json.open() as f:
        src_data = json.load(f)

    converted = {
        'info': {
            'description': 'SoccerNet 4K challenge metadata',
            'version': '1.0',
        },
        'licenses': [],
        'images': convert_images(
            src_images=src_data['images'],
            target_width=3840,
            target_height=2160,
        ),
        'annotations': [],
        'categories': src_data['categories'],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(converted))
    print(f'Wrote {output_json}')


if __name__ == '__main__':
    main()
