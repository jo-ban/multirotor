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
예측 NPZ/PNG, 평가 CSV/PNG를 Drive의 지정된 고정 폴더에 저장하고 같은 이름의 파일을 덮어씁니다. 날짜별 결과 폴더는 만들지 않습니다.
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
예시의 ground_z는 확인되지 않은 값이므로 비워 두었습니다. 실제 값으로 채워야 합니다.
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

`predict_nd.py`의 물리 조건은 `--rd`, `--l-over-d`,
`--rotor-z`, `--ground-z`, `--z-max`로 전달합니다. `--help`로 각 스크립트의 옵션을 확인하세요.
평가는 `type=V` 케이스가 있을 때 실행합니다. 해당 케이스는 학습에서 제외됩니다.
모델 입력은 **L/D, sin(theta), cos(theta), z/R_D의 4채널**입니다.
추출 범위는 `ground_z`부터 실제 메시 최대 z까지이며 z좌표는 아래에서 위로 증가합니다.
예측의 --ground-z, --z-max에는 실제 값을 반드시 지정하세요.
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
`bash rotor.sh predict --rd 9.3101751 --l-over-d 1.6 --ground-z <실제값> --z-max <실제값>`처럼 실행하면
실제 z축 Plot과 마우스 속도 표시가 나타납니다. <실제값>은 숫자로 바꿔야 합니다.
Tk 및 디스플레이 연결이 필요합니다. Colab에서는 `--no-show --plot-output ...`를 사용합니다.
자세한 설치·실행 명령은 [Linux 안내](work_multirotor/README_LINUX.md)를 참고하세요.

예측 그래프와 추출 데이터 확인 그래프는 속도를 원통좌표
`U_r`(반경), `U_theta`(방위각), `U_z`(축) 성분으로 표시합니다. 예측 NPZ에는
`u_r_mps`, `u_theta_mps`, `u_z_mps`가 저장되며, 기존 코드 호환을 위해
Cartesian `u_mps`, `v_mps`, `w_mps`도 함께 유지됩니다.

## 총 추력 고정 및 DL 자동 계산

모든 case의 총 추력은 30,000 N입니다. 동일 직경의 로터 4개가 균등하게
추력을 분담하므로 로터당 7,500 N이며, 개별 DL = 30000 / (π D²) [N/m²]입니다.
D는 RD와 L/D에서 역산합니다. 외접원 면적이 아닌 개별 로터 디스크 면적을 사용합니다.
CSV에서 load, disk_loading, DL 열은 필요 없으며, 기존 열이 남아 있어도 무시합니다.
예측도 요청한 L/D와 RD에서 DL을 다시 계산합니다. 기존 --disk-loading 인자는
이전 노트북 호환용으로만 받으며 무시합니다. 계산된 DL은 추출·예측 NPZ에 저장됩니다.

기존 데이터를 다른 DL로 무차원화했다면 새 출력 폴더로 다시 추출하고 재학습하세요.
이미 저장된 데이터나 모델의 스케일은 자동으로 변경되지 않습니다.

## CFD·예측·오차 비교 HTML

[최신 GitHub 노트북을 Colab에서 열기](https://colab.research.google.com/github/jo-ban/multirotor/blob/main/multirotor_colab.ipynb)

검증 실행은 학습에서 제외된 `type=V`의 실제 추출값과 같은 조건·좌표의
모델 예측값을 m/s로 복원한 뒤 비교합니다. 예측 실행과 검증 실행은 독립적입니다.
이미 호환되는 추출 데이터와 학습 모델이 있다면 시각화 추가 때문에 다시 학습할 필요는 없습니다.

- 예측: 기존 2D PNG와 회전 HTML을 유지하며 Colab에 HTML 다운로드 버튼을 표시합니다.
- 검증: 케이스별 `comparison_<case_id>.html`, 원본 비교 배열 `.npz`,
  네 물리량의 4행 × 3열 `error_<case_id>.png`, 기존 MAE CSV를 저장합니다.
- 비교 HTML은 U_r, U_theta, U_z, 속력 버튼으로 성분을 선택하고 CFD / 예측 / 절대오차 원통 3개를 표시합니다.
  기본은 U_r이며 선택한 성분의 MAE/NMAE와 A/B 주석·마커만 표시합니다.
  최상위 FATO / SA 영역 버튼 아래에 성분 버튼이 있습니다. SA 데이터가 없으면 `SA 데이터 없음`으로 안내합니다.
  마우스로 회전·확대하고 위치별 m/s를 확인할 수 있습니다.
- Colab에서는 검증 케이스 선택 및 회전 3D / 전개면 2D 전환이 가능합니다.
  HTML 다운로드는 선택 사항이며 결과는 Drive에도 저장됩니다.
- CFD와 예측의 색상 범위는 물리량별로 같고, 오차는 0부터 시작하는 별도 범위입니다.
  속력 오차는 속력 차이의 절대값이며 속도 오차 벡터의 크기와 다릅니다.
- 측정 위치는 반경 2RD의 원통 표면입니다. 각도에 따른 실제 분포를 유지하며
  원통 내부 체적이나 축대칭 복제 데이터를 생성하지 않습니다.
  x/y는 원통 중심에 대한 상대 좌표, z는 추출된 실제 높이입니다.
- HTML의 3D 표시는 성능을 위해 최대 64개 높이 × 96개 각도를 선택합니다.
  MAE, 최대 오차 통계, 2D 플롯에는 RD 이하의 전체 원본 격자를 사용합니다.
  비교 NPZ는 전체 높이의 원본 배열을 보존합니다.
  원통 이음새의 반복 열은 표시용이며 오차 통계에는 포함하지 않습니다.
- HTML에는 Plotly를 포함하므로 다운로드 후 인터넷 없이 열 수 있습니다.
  브라우저의 WebGL 자원이 부족하면 Colab의 전개면 2D를 사용하세요.

기존 Drive 사본 노트북은 GitHub 코드를 갱신해도 셀 내용까지 자동 갱신되지는 않습니다.
위의 최신 GitHub 노트북을 열고 기존 Drive 경로를 설정하세요.
기존 데이터·모델로 검증만 할 경우 DATASET, MODELS, CSV, RESULTS를 해당 저장 경로로
설정하고 검증 셀만 실행할 수 있습니다. 학습 셀을 다시 실행할 필요는 없습니다.

## RD 표시 범위와 최대 오차

RD는 기존 `disk_radius_m` 외접 반경[m]이며 기본값은 9.3101751 m입니다.
예측/검증 그래프는 기존 절대 z좌표를 유지하고 상단을 `min(RD, 데이터 최대 z)`로 제한합니다.
하단은 기존 데이터 최소 z입니다. 기존 비교 NPZ에 RD가 없으면 `cylinder_radius_m / 2`로 복원합니다.
학습·추출·예측 격자와 원본 NPZ는 자르지 않습니다.

검증 그래프의 U_r, U_theta, U_z, 속력마다 A는 최대 절대오차[m/s], B는 최대 위치별 오차율[%]입니다.
각 주석에 theta[deg], x/y/z[m]와 값을 표시하며 x/y는 원통 중심 기준입니다.
마커는 원본 격자에서 찾기 때문에 3D 축소 표시에서 생략되는 지점도 정확히 표시합니다.
주석은 그래프 오른쪽 별도 영역에 배치합니다. 동률이면 z, theta 순서의 첫 지점을 선택합니다.

기존 NMAE = MAE / mean(|CFD|) × 100은 표시 구간에서 유지합니다.
위치별 오차율 = |예측−CFD| / |CFD| × 100이며 기준값이 정확히 0인 지점은 오차율에서 제외합니다.
제외 개수를 표시하고, 전부 0이면 오차율은 N/A입니다. 절대오차에는 0 기준값도 포함합니다.
CSV에는 표시 범위, 최대값, 각 발생 좌표 및 제외 개수가 저장됩니다.

기존 비교 NPZ로 HTML·PNG·오차 CSV를 갱신하려면 Colab 마지막 RD 재생성 셀만 실행하거나:

```bash
python work_multirotor/evaluate_error.py --replot --output-dir /path/to/results/evaluation
```

재추출·재학습·모델 추론이 필요 없습니다. `--html-only`는 계속 HTML만 갱신합니다.

## Colab 데이터셋 검사

캐시는 CSV의 전체 folder 목록, 케이스 수, U/V/W 배열, 형상·좌표·자동 DL이 모두 일치할 때만 재사용합니다. 한 케이스만 저장된 불완전 캐시는 재추출합니다. 추출 로그에는 [현재/전체] 폴더명과 최종 완료 개수가 표시됩니다. CSV에 등록되지 않은 원본 폴더는 추출하지 않습니다. CFD 원본을 수정한 경우 REBUILD_DATASET=True로 설정하세요. 동일 캐시 폴더를 갱신하고 반복 실행해도 FileExistsError가 발생하지 않습니다.

## FATO / SA 표시 기반 (단계 1)

FATO는 반경 2RD, SA(Safety Area)는 별도의 반경 2.5RD 원통 격자입니다.
반경 상수와 리스트는 `data_utils_cylindrical.py` 한 곳에 정의합니다.
현재 추출·학습·예측·평가 파이프라인은 FATO만 지원하며 SA 데이터·모델을 생성하지 않습니다.
뷰어 API의 선택 인자 `sa_data`(검증), `sa_path`(예측)에 독립된 SA 데이터를 전달하면
같은 영역/성분 UI로 볼 수 있습니다. 검증 `sa_data`는 `load_comparison`과 같은 키를 사용합니다.
두 HTML 내보내기 함수의 `sa_input_path`도 독립된 SA NPZ를 받습니다.
SA의 RD 메타데이터가 없으면 반경/SA 반경비로 복원하며, FATO의 기존 반경/2 규칙은 유지합니다.
SA를 FATO 배열에서 계산하거나 추론하지 않습니다. SA 추출·별도 모델·노트북 단계는 사용자 승인 후 진행합니다.

HTML 갱신은 기존 `evaluate_error.py --html-only --output-dir .../results/evaluation`을 사용하면 됩니다.
`--replot`도 기존 이름의 HTML/PNG/통계 CSV를 갱신합니다. CSV 입력·case_id와 결과 경로는 변경하지 않습니다.
