# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for vylor-estimate
# Builds a single-file standalone binary with no external runtime dependencies.

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['src/vylor_estimator/cli.py'],
    pathex=[str(Path('src').resolve())],
    binaries=[],
    datas=[],
    hiddenimports=[
        'vylor_estimator',
        'vylor_estimator.classifier',
        'vylor_estimator.discovery',
        'vylor_estimator.parser',
        'vylor_estimator.pricing',
        'vylor_estimator.report',
        'vylor_estimator.report.base',
        'vylor_estimator.report.terminal',
        'vylor_estimator.savings',
        'vylor_estimator.service',
        'rich.markup',
        'rich.panel',
        'rich.table',
        'rich.console',
        'rich.text',
        'rich.progress',
        'rich.live',
        'rich.rule',
        'click.core',
        'click.decorators',
        'click.exceptions',
        'click.types',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'unittest', '_pytest', 'ruff'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='vylor-estimate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
