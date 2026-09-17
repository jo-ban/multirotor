# multirotor
U-Net 아키텍처 기반의 멀티로터 유동장 무차원화 학습

## Google Colab에서 실행

[Colab 노트북 열기](https://colab.research.google.com/github/jo-ban/multirotor/blob/main/multirotor_colab.ipynb)

노트북의 `DRIVE_DATA`, `DRIVE_CSV`, `DRIVE_RESULTS`를 실제 Google Drive 경로로
설정하고 위에서 아래로 실행합니다. GitHub 코드 안에 개인 Drive 경로를 넣을 필요가 없습니다.
원본을 `/content`로 복사해 추출·무차원화·학습하고, 데이터셋과 모델 3개(U/V/W),
예측 NPZ/PNG, 평가 CSV/PNG를 Drive의 실행별 폴더에 저장합니다.
Colab의 로컬 작업 공간은 임시이며 실제 데이터가 메모리와 디스크에 들어가야 합니다.

원본 폴더 예시:

```text
multirotor_data/
  cases.csv
  cases/
    case01/
      constant/polyMesh/  # points, faces, owner, neighbour, boundary 등
      1000/U             # 실제 시간 폴더와 속도장
```

CSV의 `folder`에는 `cases/case01`처럼 데이터 루트 기준 상대경로를 적습니다.
Windows의 `C:\...` 경로는 Colab에서 사용할 수 없습니다. `001` 같은 폴더명은
문자열 그대로 유지됩니다. CSV 형식은 [예시](work_multirotor/cases.example.csv)를 참고하세요.
`case.csv`라는 이름도 `DRIVE_CSV`에 지정하면 됩니다.

추출은 PyVista/VTK의 OpenFOAMReader를 사용하며 OpenFOAM 실행 프로그램을 호출하지 않습니다.
기존 OpenFOAM 폴더 구조를 유지하고, 분할된 processor 결과는 먼저 통합한 케이스를 사용하세요.
`normalization.py`의 무차원화는 추출 과정에서 호출됩니다.

## 경로를 직접 지정하는 실행 방법

저장소 루트에서 실행하는 예시입니다. 모든 경로 인자는 공백이 있으면 따옴표로 감쌉니다.

```bash
python work_multirotor/extract_nd.py --csv /content/raw/cases.csv --data-root /content/raw --output-root /content/dataset_nd
python work_multirotor/train_nd.py --csv /content/raw/cases.csv --data-root /content/dataset_nd --model-dir /content/models --epochs 100
python work_multirotor/predict_nd.py --model-dir /content/models --output /content/results/prediction.npz --no-show
python work_multirotor/evaluate_error.py --csv /content/raw/cases.csv --data-root /content/dataset_nd --model-dir /content/models --output-dir /content/results/evaluation --no-show
```

`predict_nd.py`의 물리 조건은 `--rotor-spacing`, `--rotor-diameter`, `--disk-loading`,
`--rotor-z`, `--ground-z`로 전달합니다. `--help`로 각 스크립트의 옵션을 확인하세요.
평가는 `type=V` 케이스가 있을 때 실행합니다. 해당 케이스는 학습에서 제외됩니다.
모델은 기존과 동일하게 5채널 입력의 원통표면 U-Net을 사용합니다.
