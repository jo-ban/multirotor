# Rotor: 고정 RD와 L/D 입력 — Linux 실행 안내

모델 입력은 L/D, sin(theta), cos(theta), z/R_D의 4채널입니다.
ground_z부터 메시 최대 z까지 추출하며 배열 첫 행은 지면, 마지막 행은 메시 상단입니다.
Plot의 z[m]와 커서 속도 표시도 같은 방향입니다. s 좌표로 되돌리지 않습니다.

## 설치

Linux의 코드 폴더에서 실행합니다. 코드가 ~/Openfoam/CH47D_single/Rotor에 있다면 그 위치로 이동하세요.

```bash
cd ~/Openfoam/CH47D_single/Rotor
sudo apt-get install python3 python3-venv python3-pip python3-tk
bash rotor.sh setup
bash rotor.sh check
```

Ubuntu/Debian 설치 예시입니다. 다른 배포판에서는 해당 패키지 관리자를 사용하세요.
setup은 requirements.txt의 numpy, pandas, torch, matplotlib, pyvista를 .venv에 설치합니다.
기존 환경을 쓰려면 ROTOR_PYTHON=/absolute/path/to/python을 지정하세요.
CUDA가 사용 가능하면 GPU, 아니면 CPU로 학습합니다.

## CSV와 고정 형상

기본 RD는 9.3101751 m입니다. CSV에 RD를 명시하거나 생략할 수 있습니다.
RD 별칭은 R_D, rd_m, disk_radius_m이고, 간격비는 L_over_D, L/D, l_over_d를 지원합니다.
같은 데이터셋에서는 모든 케이스의 RD가 같아야 합니다.

```csv
folder,RD,L_over_D,rotor_z,ground_z,load,type,center_x,center_y
cases/case01,9.3101751,1.5,9.3101752,,,T,0,0
cases/case02,9.3101751,1.6,9.3101752,,,V,0,0
```

위의 빈 ground_z와 load는 반드시 실제 값으로 채우세요. folder도 실제 케이스 위치로 바꾸세요.
cases.example.csv를 참고해 cases.csv를 준비합니다. RD와 L/D만으로 CFD 추출에 필요한
지면·로터면 좌표와 디스크 로딩까지 알 수 있는 것은 아닙니다.
rotor_z=9.3101752는 이전 두 로그의 로터 영역 중앙값입니다. 좌표를 이동했다면 실제 값을 쓰세요.
ground_z, load, 메시 상단은 이전 로그 분석에서 확정되지 않았으므로 예시값으로 대체하지 않습니다.

q=L/D라 할 때 다음 식으로 L과 D를 역산합니다.

    D = RD / (q/sqrt(2) + 1/2)
    L = q * D

L/D=1.5이면 L≈8.9483046 m, D≈5.9655364 m이며,
L/D=1.6이면 L≈9.1311428 m, D≈5.7069642 m입니다.
두 경우 모두 RD=9.3101751 m, 추출 원통 반경 2RD=18.6203502 m입니다.
기존 L,D 열도 RD와 일치하는 실제 값이면 읽을 수 있으나, 충돌하면 오류로 중단합니다.

folder는 --data-root 기준 상대경로 또는 Linux 절대경로입니다. center_x/y는 생략 시 0입니다.
CFD 케이스에는 mesh와 U 결과가 있어야 합니다. 최신 시간을 읽고 필요한 .foam 파일을 만듭니다.
분할된 processor 결과는 먼저 통합하세요. 추출기가 CFD 계산 자체를 실행하지는 않습니다.

## 실행

```bash
bash rotor.sh extract --data-root /data/OpenFOAM
bash rotor.sh visualize
bash rotor.sh train --epochs 100 --batch-size 4
bash rotor.sh evaluate
bash rotor.sh predict --help
```

예측에서는 아래 <실제값>을 해당 숫자로 바꾸세요. 괄호까지 입력하는 명령은 아닙니다.

```text
bash rotor.sh predict --rd 9.3101751 --l-over-d 1.6 --rotor-z 9.3101752 --ground-z <실제지면z> --z-max <메시최대z> --disk-loading <실제디스크로딩>
```

--rd 기본값은 9.3101751, --l-over-d 기본값은 1.5입니다.
--ground-z, --z-max, --disk-loading은 필수입니다. z-max는 추출 로그/NPZ에서 확인하세요.
추출에서는 메시 상단을 자동으로 읽으므로 CSV에 z-max 열을 적지 않습니다.
CSV의 rotor_z는 추출 상한이 아닙니다. 좌표 원점을 임의로 이동하거나 W 부호를 바꾸지 않습니다.
원통 일부가 CFD 영역 밖이면 오류로 중단합니다.

| 명령 | 기본 출력 |
| --- | --- |
| extract | dataset_zrd/inputs 및 targets_u/v/w |
| train | models_zrd/의 U/V/W 체크포인트 |
| visualize | results/visualize/PNG |
| evaluate | results/evaluate/CSV 및 PNG |
| predict | 대화형 Plot과 results/prediction_cylinder_r2RD.npz |

## 학습에서 CSV의 역할

학습 입력과 정답은 추출된 NPZ/NPY에서 읽습니다. CSV의 물리값을 다시 계산하지 않습니다.
CSV는 type=V인 케이스를 식별해 전체 추출 데이터에서 제외하는 분할표입니다.
즉 학습 대상은 '전체 추출 케이스 - V 케이스'입니다. CSV에 없는 추출 케이스도 포함됩니다.
서로 다른 실행의 불필요한 데이터를 같은 dataset_zrd 폴더에 섞지 마세요.

추출 후에는 별도 split.csv를 다음처럼 만들 수 있습니다.

```csv
folder,type
cases/case02,V
```

```bash
bash rotor.sh train --csv split.csv
bash rotor.sh evaluate --csv split.csv
```

folder로 추출 메타데이터의 source_folder를 찾습니다. 같은 folder의 추출 결과가 여러 개면
case_id 열에 입력 NPZ 파일명의 input_와 .npz를 뺀 ID를 지정하세요.
식별되지 않는 V 케이스와 중복 식별자는 오류로 처리합니다. V가 없으면 모두 학습하며 평가할 수 없습니다.
학습은 NPZ의 RD가 모두 같은지 확인하고 체크포인트에 RD를 저장합니다.
예측/평가 시 저장된 RD와 요청 RD가 다르면 오류로 중단합니다.

## Plot 라이브러리와 원격 실행

Matplotlib의 imshow, annotate, mpl_connect('motion_notify_event'), show를 사용합니다.
마우스를 올리면 theta, 실제 z[m], 원통좌표 속도 U_r/U_theta/U_z와 속도 크기[m/s]가 표시됩니다.
예측 NPZ의 `u_r_mps`, `u_theta_mps`, `u_z_mps`는 각각 반경, 방위각, z축 방향 속도입니다. 기존 `u_mps`, `v_mps`, `w_mps`는 후방 호환을 위해 함께 저장됩니다.
기본 GUI 백엔드는 TkAgg입니다. python3-tk는 시스템 패키지이며 pip install tkinter를 사용하지 않습니다.
mplcursors는 필요 없습니다. Qt를 쓰려면 .venv/bin/python -m pip install PyQt6 후 --backend QtAgg를 추가하세요.

```bash
.venv/bin/python -m tkinter
```

Linux 데스크톱 터미널이나 X11 forwarding이 설정된 ssh -X 세션에서 창을 열 수 있습니다.
IDE는 필요 없지만 디스플레이 연결은 필요합니다. 텍스트 전용 세션에서는 예측 명령에 --no-show를 추가하세요.
PNG 저장은 --plot-output results/prediction.png를 명시할 때만 수행합니다.
Colab 노트북은 --no-show와 --plot-output을 사용해 결과를 표시합니다.

## 경로와 검증

기본 경로는 rotor.sh가 있는 폴더 기준, 사용자가 지정한 상대경로는 현재 터미널 기준입니다.
공백이 있는 경로는 따옴표로 감싸세요. 재실행하면 같은 이름의 결과는 덮어씁니다.

```bash
nohup bash rotor.sh train --epochs 100 > train.log 2>&1 &
tail -f train.log
bash rotor.sh test
```

테스트는 합성 OpenFOAM 추출→학습→예측→평가, RD 고정 역산, V 분리, z 배열 방향과 커서 표시를 확인합니다.
실제 연구 데이터 정확도나 Linux GUI 환경의 검증을 대신하지는 않습니다.
입력 인터페이스만 바뀌었으므로 같은 RD/좌표/조건으로 추출된 z/R_D 데이터는 재사용 가능합니다.
D=1 등 잘못된 치수로 추출했다면 새 폴더로 재추출·재학습해야 합니다.
이전 s/R_D 또는 5채널 데이터·가중치는 호환되지 않습니다.
