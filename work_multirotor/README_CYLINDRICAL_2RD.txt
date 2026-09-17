[멀티로터 r=2R_D 원통 표면 예측 버전]

목표
- 멀티로터 시스템 중심에서 반경 r=2R_D인 원통 옆면의 속도장을 추출합니다.
- 4개의 동일한 로터가 정사각형으로 배치된 구성을 사용합니다.
- L은 인접한 개별 로터 중심 사이 거리, D는 개별 로터 한 개의 직경입니다.
- R_D는 4개 로터 전체(블레이드 끝 포함)를 감싸는 외접원의 반경입니다.
- 코드가 R_D = L/sqrt(2) + D/2로 자동 계산합니다.
- 추출 원통 반경은 2R_D = sqrt(2)L + D입니다.
- 원통 표면을 가로=방위각 theta, 세로=s/R_D인 2D 평면으로 펼칩니다.
- s는 로터면에서 지면 방향으로 잰 거리이며, s=0은 로터면입니다.
- L/D를 케이스 변수로 사용하고 U, V, W를 각각 별도 2D U-Net으로 학습합니다.

출력 배열
- shape: (Nz, Ntheta), 기본값 (256, 256)
- 가로축: theta = 0~360 deg (360도 끝점은 중복 저장하지 않음)
- 세로축: s/R_D=0~H/R_D (로터면에서 지면까지)
- U, V, W는 유도속도 Vi로 무차원화되어 저장됩니다.
- 케이스 ID에는 L/D, R_D, H/R_D가 포함되어 높이가 다른 데이터가 덮어써지지 않습니다.

네트워크 입력 채널
- CH0: L/D
- CH1: H/R_D (로터면과 지면 사이 거리/외접반경)
- CH2: sin(theta)
- CH3: cos(theta)
- CH4: s/R_D (로터면에서 지면 방향으로 잰 거리)

주기 경계
- theta=0도와 360도는 같은 위치이므로 합성곱의 theta 방향에 circular padding을 적용했습니다.

cases.csv 필수 열
- folder: OpenFOAM 케이스 폴더
- L: 인접한 개별 로터 중심 사이 거리[m]
  (spacing, rotor_spacing, rotor_spacing_m도 허용)
- D: 개별 로터 한 개의 직경[m]
  (diameter, rotor_diameter, rotor_diameter_m도 허용)
- rotor_z: OpenFOAM 좌표계에서 로터면의 실제 z 좌표[m]
- ground_z: OpenFOAM 좌표계에서 지면의 실제 z 좌표[m]
- L/D와 R_D는 L, D로부터 코드가 자동 계산합니다.
- load: 디스크 로딩[N/m^2] (disk_loading, DL도 허용)

cases.csv 선택 열
- type: V이면 검증용으로 분리
- center_x, center_y: 멀티로터 시스템 중심[m], 생략 시 (0, 0)

예시
folder,L,D,rotor_z,ground_z,load,type,center_x,center_y
case_100,1.00,1.00,2.00,0.00,153.22,T,0.0,0.0
case_125,1.25,1.00,2.00,0.00,153.22,T,0.0,0.0
case_150,1.50,1.00,2.00,0.00,153.22,V,0.0,0.0

설정
- extract_nd.py의 CYLINDER_RADIUS_OVER_RD = 2.0
- extract_nd.py의 RESOLUTION_Z_THETA = (256, 256)
- 높이 범위는 cases.csv의 rotor_z에서 ground_z까지 자동 설정됩니다.
- H=|rotor_z-ground_z|, 세로축 범위는 s/R_D=0~H/R_D입니다.

실행 순서
1. python extract_nd.py
2. python visualize_cylindrical_data.py   (선택)
3. python train_nd.py
4. python evaluate_error.py               (type=V가 있을 때)
5. python predict_nd.py

주의
- 로터 4개, 정사각형 배치, 회전 방향, 직경, 디스크 로딩과 계산 조건은 고정하고 L/D만 바꾸는 구성을 전제로 합니다.
- 원통 r=2R_D가 OpenFOAM 계산영역 안에 전부 포함되어야 합니다.
- 기존 3D 모델 파일과 호환되지 않으므로 새로 학습해야 합니다.
- 기존 4채널 원통 모델과도 호환되지 않으며 5채널 모델을 새로 학습해야 합니다.
