"""Validate every CSV case before reusing an extracted dataset."""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from data_utils_cylindrical import (
    center_from_row, disk_loading_from_row, geometry_from_row,
    load_case_input, rotor_ground_z_from_row, CYLINDER_RADIUS_OVER_RD,
)


def dataset_status(root, csv_path, radius_over_rd=CYLINDER_RADIUS_OVER_RD):
    root = Path(root)
    frame = pd.read_csv(csv_path, dtype={'folder': str})
    if 'folder' not in frame or frame.empty:
        return False, 'CSV에 folder 행이 없습니다.'
    folders = frame['folder'].fillna('').str.strip().str.replace('\\', '/', regex=False)
    if (folders == '').any() or folders.duplicated().any():
        return False, 'CSV folder에 빈 값 또는 중복이 있습니다.'
    paths = sorted((root / 'inputs').glob('input_*.npz'))
    if len(paths) != len(frame):
        return False, f'CSV {len(frame)}개 / 저장된 입력 {len(paths)}개: 개수 불일치'
    records = {}
    try:
        for path in paths:
            meta = load_case_input(path)
            if not np.isclose(meta['radius_over_rd'], radius_over_rd, rtol=1e-6):
                return False, f'{path.name}: 영역 반경이 다릅니다.'
            folder = meta['source_folder']
            if folder in records:
                return False, f'중복 추출 케이스: {folder}'
            case_id = path.stem.removeprefix('input_')
            for component in ('u', 'v', 'w'):
                target = root / f'targets_{component}' / f'{component}_{case_id}.npy'
                array = np.load(target, mmap_mode='r')
                if array.shape != meta['shape_ztheta']:
                    return False, f'배열 크기 불일치: {target.name}'
                del array
            records[folder] = meta
        if set(records) != set(folders):
            return False, f'CSV와 추출 폴더 불일치: 누락 {sorted(set(folders) - set(records))}'
        for (_, row), folder in zip(frame.iterrows(), folders):
            meta = records[folder]
            rd, ratio, _, diameter = geometry_from_row(row)
            rotor_z, ground_z = rotor_ground_z_from_row(row)
            actual = [meta[k] for k in ('disk_radius_m', 'l_over_d', 'rotor_diameter_m',
                                       'rotor_z_m', 'ground_z_m', 'disk_loading')]
            expected = [rd, ratio, diameter, rotor_z, ground_z, disk_loading_from_row(row)]
            if not np.allclose(actual, expected, rtol=1e-6, atol=1e-7):
                return False, f'{folder}: 형상/좌표/DL이 현재 CSV·총 추력 기준과 다릅니다.'
            if not np.allclose(meta['center_xy_m'], center_from_row(row), rtol=1e-6, atol=1e-7):
                return False, f'{folder}: 중심 좌표가 다릅니다.'
    except (OSError, ValueError, KeyError, EOFError) as exc:
        return False, f'추출 파일 검사 실패: {exc}'
    return True, f'CSV {len(frame)}개 / 추출 {len(records)}개 / U,V,W 확인 완료'


def sync_dataset(source, destination):
    """Reuse a fixed directory; replace generated files without mixing old cases."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        return
    for subdir, pattern in [('inputs', 'input_*.npz')] + [
            (f'targets_{c}', f'{c}_*.npy') for c in ('u', 'v', 'w')]:
        target = destination / subdir
        target.mkdir(parents=True, exist_ok=True)
        files = list((source / subdir).glob(pattern))
        names = {p.name for p in files}
        if not files:
            raise ValueError(f'Empty source dataset: {source / subdir}')
        for path in files:
            temporary = target / (path.name + '.tmp')
            shutil.copy2(path, temporary)
            temporary.replace(target / path.name)
        for path in target.glob(pattern):
            if path.name not in names:
                path.unlink()
