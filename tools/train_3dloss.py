#!/usr/bin/env python3
"""Prepends pose-workspace-private/ViTPose to sys.path so its local mmpose
(with LocSim3DLoss and TopDown 3D-loss support) is loaded instead of any
other installed mmpose, then delegates to the original tools/train.py."""
import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_TOOLS_DIR = os.path.join(_REPO_ROOT, 'tools')
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from train import main

main()
