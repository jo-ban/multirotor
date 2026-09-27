# multirotor
U-Net 아키텍처 기반의 멀티로터 유동장 무차원화 학습

## 회전 가능한 3D 결과

Colab 예측 셀은 기존 NPZ/PNG와 함께 `results/prediction_3d.html`을 Drive에 저장하고
노트북에도 3D 그래프를 표시합니다. HTML을 내려받아 브라우저에서 열면 Colab 연결 없이
회전·확대·값 확인 및 U_r/U_theta/U_z/속력 선택을 할 수 있습니다.
`predict_nd.py` 자체가 NPZ 옆에 `<NPZ 이름>_3d.html`을 자동 생성합니다.
기존 Colab 노트북도 GitHub 코드와 requirements를 업데이트하면 예측 셀 수정 없이 HTML이 저장됩니다.
이미 실행 중인 Colab은 저장소 코드를 갱신하고 requirements를 다시 설치한 뒤 예측 셀을 실행하세요.

기존 예측 결과만으로도 생성할 수 있습니다(학습/예측 재실행 불필요).

```bash
pip install -r work_multirotor/requirements.txt
python work_multirotor/visualize_prediction_3d.py --input prediction.npz --output prediction_3d.html
```

Linux 런처에서는 `bash rotor.sh view3d --input prediction.npz --output prediction_3d.html`을 사용합니다.
색은 실제 저장된 속도[m/s]이며 성분은 0 중심 대칭 색 범위, 속력은 0 이상 범위를 사용합니다.
측정면은 고정 반경 2RD의 원통이며 내부 체적 유동장을 의미하지 않습니다.
중앙 로터는 저장된 간격/직경/높이를 이용한 정사각형 배치 도식이고 실제 기체 CAD가 아닙니다.
원통과 로터의 수평 중심은 (0, 0), 높이는 NPZ의 실제 z 좌표입니다.

## Google Colab에서 실행

[Google Drive Colab 노트북 열기](https://colab.research.google.com/drive/183B-S7CGxb6Yrkflm3Nd-maWbtMf7OQ1?authuser=2)

이 노트북은 실행 시 `https://github.com/jo-ban/multirotor.git`을 clone하여
GitHub의 최신 코드를 사용합니다.

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
예시의 ground_z와 load는 확인되지 않은 값이므로 비워 두었습니다. 실제 값으로 채워야 합니다.
`case.csv`라는 이름도 `DRIVE_CSV`에 지정하면 됩니다.

추출은 PyVista/VTK의 OpenFOAMReader를 사용하며 OpenFOAM 실행 프로그램을 호출하지 않습니다.
기존 OpenFOAM 폴더 구조를 유지하고, 분할된 processor 결과는 먼저 통합한 케이스를 사용하세요.
`normalization.py`의 무차원화는 추출 과정에서 호출됩니다.

## 경로를 직접 지정하는 실행 방법

저장소 루트에서 실행하는 예시입니다. 모든 경로 인자는 공백이 있으면 따옴표로 감쌉니다.

```bash
python work_multirotor/extract_nd.py --csv /content/raw/cases.csv --data-root /content/raw --output-root /content/dataset_nd
python work_multirotor/train_nd.py --csv /content/raw/cases.csv --data-root /content/dataset_nd --model-dir /content/models --epochs 100
python work_multirotor/evaluate_error.py --csv /content/raw/cases.csv --data-root /content/dataset_nd --model-dir /content/models --output-dir /content/results/evaluation --no-show
```

`predict_nd.py`의 물리 조건은 `--rd`, `--l-over-d`, `--disk-loading`,
`--rotor-z`, `--ground-z`, `--z-max`로 전달합니다. `--help`로 각 스크립트의 옵션을 확인하세요.
평가는 `type=V` 케이스가 있을 때 실행합니다. 해당 케이스는 학습에서 제외됩니다.
모델 입력은 **L/D, sin(theta), cos(theta), z/R_D의 4채널**입니다.
추출 범위는 `ground_z`부터 실제 메시 최대 z까지이며 z좌표는 아래에서 위로 증가합니다.
예측의 --ground-z, --z-max, --disk-loading에는 실제 값을 반드시 지정하세요.
Colab은 선택한 추출 NPZ에서 해당 값과 rotor_z를 읽습니다. 기본 기준은 첫 추출 케이스이며
REFERENCE_INPUT_NAME으로 변경할 수 있습니다. 임의의 z_max=3 또는 D=1을 기본값으로 사용하지 않습니다.
이전 s/R_D 또는 5채널 모델은 재사용할 수 없습니다. 이미 올바른 RD로 추출한 z/R_D 데이터는 재사용 가능합니다.

## 고정 RD와 CSV 입력

RD는 기본 9.3101751 m이며 L/D가 바뀌면 L과 D를 함께 역산합니다.
q=L/D에 대해 D=RD/(q/sqrt(2)+1/2), L=q*D입니다. 추출 반경은 2RD입니다.
CSV는 RD,L_over_D 열을 받습니다(R_D, L/D 별칭도 지원). RD를 생략하면 기본값을 사용합니다.
한 데이터셋의 RD는 동일해야 합니다. L,D가 추가로 있다면 역산값과 충돌할 때 오류로 중단합니다.

학습 입력은 추출 NPZ에서 읽고, CSV는 V 케이스를 식별하는 데 사용합니다.
전체 추출 데이터에서 V 케이스를 제외하며 folder,type 또는 case_id,type만 있는 분할표도 지원합니다.
CSV의 나머지 물리값을 학습 단계에서 다시 적용하지 않습니다.

## Linux 터미널과 대화형 Plot

`cd work_multirotor` 후 `bash rotor.sh setup`으로 설치합니다.
`bash rotor.sh predict --rd 9.3101751 --l-over-d 1.6 --ground-z <실제값> --z-max <실제값> --disk-loading <실제값>`처럼 실행하면
실제 z축 Plot과 마우스 속도 표시가 나타납니다. <실제값>은 숫자로 바꿔야 합니다.
Tk 및 디스플레이 연결이 필요합니다. Colab에서는 `--no-show --plot-output ...`를 사용합니다.
자세한 설치·실행 명령은 [Linux 안내](work_multirotor/README_LINUX.md)를 참고하세요.

예측 그래프와 추출 데이터 확인 그래프는 속도를 원통좌표
`U_r`(반경), `U_theta`(방위각), `U_z`(축) 성분으로 표시합니다. 예측 NPZ에는
`u_r_mps`, `u_theta_mps`, `u_z_mps`가 저장되며, 기존 코드 호환을 위해
Cartesian `u_mps`, `v_mps`, `w_mps`도 함께 유지됩니다.
