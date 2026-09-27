# Rotor: Linux 터미널 실행 안내

IDE 없이 Bash와 Python으로 실행하는 work_multirotor 원통 표면 모델입니다.
로컬 `C:\work_multirotor\github_colab\work_multirotor`의 CLI 버전을 기반으로
작성했습니다. GitHub 최신 원격 코드와의 동기화는 수행하지 않았습니다.

## 1. Linux로 복사하고 설치

이 Rotor 폴더 전체를 Linux의 `~/Rotor` 등에 복사하세요. Windows의 `C:\Rotor`는
Linux 경로가 아니므로 Linux에서는 실제 복사한 경로를 사용합니다.
아래 예시는 `~/Rotor`에 복사한 경우입니다.

```bash
cd ~/Rotor
bash rotor.sh setup
bash rotor.sh check
```

Python 3와 venv/pip가 필요합니다. Ubuntu/Debian에서 없다면 관리자가 다음 명령으로
설치할 수 있습니다(다른 배포판은 해당 패키지 관리자 사용).

```bash
sudo apt-get update
sudo apt-get install python3 python3-venv python3-pip python3-tk
```

`setup`은 `.venv`에 requirements.txt를 설치하며 인터넷 접속이 필요합니다.
패키지 버전은 기존 프로젝트처럼 고정하지 않았으므로 설치 환경에 따라 달라질 수 있습니다.
인터넷이 차단된 서버에서는 관리자에게 패키지가 설치된 Python 환경을 요청하고 지정하세요.

```bash
export ROTOR_PYTHON=/absolute/path/to/environment/bin/python
bash rotor.sh check
```

가상환경 활성화나 IDE는 필요하지 않습니다. `check`에 CUDA가 False로 표시되면 CPU로
학습합니다. GPU를 사용하려면 서버의 드라이버와 호환되는 PyTorch 환경이 필요합니다.

## 2. 입력 데이터 지정

동봉된 `cases.csv`는 예시입니다. 실제 계산 조건과 OpenFOAM 결과 폴더로 수정하세요.
터미널 편집기 `nano cases.csv` 또는 `vi cases.csv`를 사용할 수 있습니다.

```csv
folder,L,D,rotor_z,ground_z,load,type,center_x,center_y
case_100,1.00,1.00,2.00,0.00,153.22,T,0.0,0.0
case_125,1.25,1.00,2.00,0.00,153.22,T,0.0,0.0
case_150,1.50,1.00,2.00,0.00,153.22,V,0.0,0.0
```

- `folder`: Linux 절대 경로 또는 데이터 루트 기준 상대 경로. `C:\...`는 사용하지 않습니다.
- `L`, `D`: 로터 중심 간 거리와 개별 로터 직경[m].
- `rotor_z`, `ground_z`: 로터면 및 지면의 실제 z 좌표[m].
- `load`: 디스크 로딩[N/m²]. `type=V`는 검증용이며 학습에서 제외됩니다.
- `center_x`, `center_y`: 시스템 중심[m].

예를 들어 `/data/OpenFOAM/case_100` 등에 데이터가 있다면 다음 명령을 사용합니다.
데이터 루트를 생략하면 CSV가 있는 폴더 기준으로 찾습니다.
각 케이스에는 OpenFOAM mesh와 시간 디렉터리의 U 결과가 있어야 합니다.
추출 시 최신 시간을 읽으며 필요하면 케이스 폴더에 `.foam` 파일을 생성합니다.
분할된 processor 결과는 먼저 합쳐서 준비하세요. 이 도구는 CFD 계산 자체를 실행하지 않습니다.

## 3. 단계별 실행 명령

```bash
cd ~/Rotor

# 1) OpenFOAM 결과에서 원통 표면 추출
bash rotor.sh extract --data-root /data/OpenFOAM

# 2) 추출 데이터 중 한 케이스를 PNG로 확인 (선택)
bash rotor.sh visualize

# 3) U, V, W 모델 학습
bash rotor.sh train --epochs 100 --batch-size 4

# 4) CSV의 type=V 케이스 평가
bash rotor.sh evaluate

# 5) 지정 조건 예측: Plot 창을 열고 마우스 위치의 속도 표시
bash rotor.sh predict --rotor-spacing 1.25 --rotor-diameter 1.0 \
  --disk-loading 153.22 --rotor-z 2.0 --ground-z 0.0 --z-max 3.0
```

평가에는 검증 데이터와 학습 모델이, 예측에는 학습 모델이 필요합니다.
모든 명령은 실패 시 0이 아닌 종료 코드를 반환합니다. 연속 실행에는 `&&`를 사용하세요.
재실행하면 같은 이름의 출력/모델은 덮어씁니다. 보존하려면 다른 출력 경로를 지정하세요.

| 명령 | 실행 코드 | 기본 출력 |
| --- | --- | --- |
| extract | extract_nd.py | dataset_zrd/ |
| visualize | visualize_cylindrical_data.py | results/visualize/surface_*.png |
| train | train_nd.py | models_zrd/rotor_unet_cyl2d_u/v/w.pth (3개 파일) |
| evaluate | evaluate_error.py | results/evaluate/의 CSV와 PNG |
| predict | predict_nd.py | 대화형 Plot 창 + results/prediction_cylinder_r2RD.npz (PNG 자동 저장 없음) |

`model_cylindrical.py`, `data_utils_cylindrical.py`, `normalization.py`는 위 코드에서
불러오는 모듈이므로 따로 실행할 필요가 없습니다. 이 모델은 기존 3D 체크포인트와
호환되지 않습니다. 물리적 가정은 `README_CYLINDRICAL_2RD.txt`를 참고하세요.

입력은 **CH0=L/D, CH1=sin(theta), CH2=cos(theta), CH3=z/R_D**의 4채널입니다.
`z`는 지면에서의 거리가 아니라 OpenFOAM의 실제 z 좌표입니다. 좌표 원점은 이동하지 않습니다.
추출 시 `internalMesh.bounds[5]`에서 최대 z를 자동으로 읽으며, `ground_z`부터 그 값까지
균등하게 샘플링합니다. `rotor_z`는 추출 상한이 아닙니다. 첫 행은 지면, 마지막 행은 메시 상단입니다.
R_D=L/sqrt(2)+D/2로 원통 반경 2R_D와 z/R_D를 계산합니다. H/R_D 상수 채널은 없습니다.

**기존 s/R_D 데이터와 4·5채널 가중치는 재사용할 수 없습니다. 새로 추출하고 학습해야 합니다.**
채널 개수가 같아도 좌표 의미와 영역이 달라졌기 때문입니다. 데이터/체크포인트에 좌표 버전을
기록하고 오래된 파일은 오류로 거부합니다. 기본 데이터·모델 폴더는 `dataset_zrd`, `models_zrd`입니다.

예측은 메시를 읽지 않으므로 **`--z-max`를 반드시 지정**합니다. 추출 로그 또는 입력 NPZ의
`z_max_m`에서 확인한 실제 메시 최대 z를 넣으세요. 예시의 `3.0`은 사용자의 메시 크기가 아닙니다.
로터면 z=0, 지면 z=-2, 메시 상단 z=3인 예:

```bash
bash rotor.sh predict --rotor-z 0 --ground-z -2 --z-max 3
```

비정형 메시에서 최대 z까지의 원통 일부가 CFD 영역 밖이면 추출을 실패 처리합니다.
유효하지 않은 점을 0으로 채워 학습하지 않습니다. CSV는 실제 지면과 같은 원점의 좌표를 사용하세요.
학습/검증 분리는 CSV의 folder와 추출 메타데이터를 대조하므로 메시 상단을 CSV에 적을 필요는 없습니다.
원본 영역을 변경한 뒤 재추출할 때에는 새 데이터 디렉터리를 지정하세요.

## 4. 경로·옵션·백그라운드 실행

```bash
bash rotor.sh help
bash rotor.sh train --help
bash rotor.sh predict --help
bash rotor.sh extract --csv /data/cases.csv --data-root /data/OpenFOAM
bash rotor.sh train --model-dir /data/new_models --epochs 50
bash rotor.sh predict --z-max 3.0 --model-dir /data/new_models --output /data/prediction.npz
nohup bash rotor.sh train --epochs 100 > train.log 2>&1 &
tail -f train.log
```

기본 입력/출력은 rotor.sh가 있는 폴더를 기준으로 합니다. 사용자가 옵션에 넣은 상대 경로는
현재 터미널 위치를 기준으로 합니다. 공백이 있는 경로는 따옴표로 감싸세요.
셸 스크립트 없이도 실행할 수 있습니다.

```bash
cd ~/Rotor
MPLBACKEND=Agg .venv/bin/python train_nd.py --csv cases.csv \
  --data-root dataset_zrd --model-dir models_zrd --epochs 100
.venv/bin/python predict_nd.py --model-dir models_zrd \
  --output results/prediction.npz --z-max 3.0
```

## 5. 설치 후 통합 테스트

```bash
bash rotor.sh test
```

테스트는 임시 폴더에 작은 합성 OpenFOAM 케이스를 만들어 추출 → 1 epoch 학습 →
시각화 → 예측 → 평가 및 누락 경로 오류를 검사합니다. 실제 연구 데이터의 정확도를
검증하는 테스트는 아닙니다. 실제 Linux/GPU 환경에서의 실행은 대상 서버에서 확인해야 합니다.

## 6. 예측 Plot 창과 마우스 속도 표시

`predict`는 기본적으로 Matplotlib 창에 U, V, W, 속도 크기를 표시합니다.
각 그림 위로 마우스를 이동하면 해당 셀의 **theta[deg], z[m], U/V/W 및 속도 크기[m/s]**가
그림 안의 정보 상자와 창 하단 상태 표시줄에 나타납니다. 값은 보간값이 아닌 표시된 셀의
예측값이며, theta와 z는 해당 셀의 원래 샘플 좌표입니다. 그림 밖에서는 정보 상자가 숨겨집니다.
창을 닫으면 명령이 종료됩니다. 수치 데이터 NPZ는 저장하지만 PNG는 자동 저장하지 않습니다.
`visualize`와 `evaluate`의 기존 PNG 출력은 유지됩니다.

### 설치할 라이브러리와 사용 함수

| 라이브러리 | 용도 / 사용 함수 | 설치 |
| --- | --- | --- |
| matplotlib | `imshow`, `annotate`, `mpl_connect('motion_notify_event', ...)`, `show(block=True)` | `bash rotor.sh setup`에 포함 |
| tkinter / Tk | 기본 `TkAgg` 그래프 창 | Ubuntu/Debian: `sudo apt-get install python3-tk` |
| numpy, torch, pandas, pyvista | 기존 예측·학습·데이터 처리 | `bash rotor.sh setup`에 포함 |

마우스 이벤트는 Matplotlib 내장 기능을 사용하므로 `mplcursors`를 추가 설치할 필요는 없습니다.
`tkinter`는 pip 패키지가 아니므로 `pip install tkinter`를 사용하지 마세요.
별도 Python/Conda를 사용한다면 그 Python에 맞는 Tk 패키지가 필요합니다.

```bash
cd ~/Rotor
sudo apt-get install python3-tk
bash rotor.sh setup
.venv/bin/python -m tkinter  # 테스트 창이 뜨면 닫기
bash rotor.sh predict --rotor-spacing 1.25 --rotor-diameter 1.0 \
  --disk-loading 153.22 --rotor-z 2.0 --ground-z 0.0 --z-max 3.0
```

이미 가상환경이 설치되어 있다면 `setup`을 다시 실행할 필요 없이 Tk 설치 후 예측하면 됩니다.
Tk 대신 Qt가 필요한 환경에서는 다음 대안을 사용합니다.

```bash
.venv/bin/python -m pip install PyQt6
bash rotor.sh predict --z-max 3.0 --backend QtAgg
```

IDE는 필요 없지만 **그래프 창을 표시할 디스플레이 연결은 필요합니다.** Linux 데스크톱의
터미널에서는 바로 실행합니다. 원격 서버의 SSH 터미널에서는 클라이언트의 X 서버와
서버의 X11 forwarding 설정이 필요합니다. 설정된 환경의 접속 예시는 다음과 같습니다.

```bash
ssh -X username@server
cd ~/Rotor
bash rotor.sh predict --z-max 3.0
```

디스플레이 연결이 없는 순수 텍스트 세션에서는 창을 띄울 수 없습니다.
`no display`, `headless`, `Cannot load backend TkAgg` 오류는 디스플레이/X11 연결을,
`No module named tkinter`는 Tk 설치를 확인하세요. 예측 창은 `nohup`으로 실행하지 마세요.
수치 결과만 필요한 경우 또는 명시적으로 PNG를 저장할 경우:

```bash
bash rotor.sh predict --z-max 3.0 --no-show
bash rotor.sh predict --z-max 3.0 --no-show --plot-output results/prediction.png
```

공식 참고: [Matplotlib GUI 백엔드](https://matplotlib.org/stable/users/explain/figure/backends.html),
[마우스 이벤트 처리](https://matplotlib.org/stable/users/explain/figure/event_handling.html).
