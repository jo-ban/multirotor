[멀티로터 고정 RD / 실제 z/R_D 4채널]

RD = 9.3101751 m 기본 고정값. 입력은 RD와 L_over_D (별칭 L/D).
RD 생략 시 기본값 사용. 같은 추출/학습 데이터셋은 하나의 RD만 사용.
q=L/D, D=RD/(q/sqrt(2)+1/2), L=q*D.
L/D 변화 시 L과 D가 함께 바뀌며 원통 추출 반경 2RD는 유지.

CH0=L/D, CH1=sin(theta), CH2=cos(theta), CH3=z/R_D.
z는 OpenFOAM 실제 좌표. 지면을 빼거나 로터면으로 이동하지 않음.
추출은 ground_z부터 internalMesh.bounds[5]까지.
배열 첫 행=지면, 마지막 행=메시 상단. Plot은 아래에서 위로 z 증가.
rotor_z는 물리 조건 메타데이터이며 추출 상한이 아님.
H/R_D 상수 입력 채널 및 s/R_D 입력은 없음.

CSV: folder,RD,L_over_D,rotor_z,ground_z,load,type,center_x,center_y
RD,L_over_D로 형상 계산. ground_z/rotor_z/load는 실제 값을 입력.
로터면 9.3101752는 이전 두 로그 분석의 중앙 높이이며 좌표를 옮겼다면 다시 확인.
예시 파일의 ground_z/load 공란은 사용자가 실제 값으로 채워야 함.

학습 입력/정답은 추출 NPZ/NPY 사용.
CSV는 folder,type (또는 case_id,type)으로 V 케이스만 구분.
전체 추출 데이터에서 V 케이스를 제외해 학습. 나머지 CSV 물리값은 학습에 사용하지 않음.

예측: --rd 9.3101751 --l-over-d 1.6
추가 필수 옵션: --ground-z, --z-max, --disk-loading (실제 값).
기본 --rotor-z는 9.3101752. Plot/커서는 z[m] 표시.
Colab은 기준 추출 NPZ에서 좌표·로딩·z_max를 읽음.

좌표/배열 규칙은 absolute_z_over_rd_v1 그대로 유지.
동일 물리조건의 z/R_D 데이터는 입력 방식 변경만으로 재추출할 필요 없음.
잘못된 RD로 추출한 데이터 또는 이전 s/R_D·5채널 데이터는 재추출·재학습 필요.
실행과 라이브러리 설치 명령은 README_LINUX.md 참고.
