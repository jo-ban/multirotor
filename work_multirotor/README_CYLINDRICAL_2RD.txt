[멀티로터 원통 표면: 실제 z/R_D 4채널 버전]

영역과 좌표
- 정사각형 4로터의 외접반경: R_D = L/sqrt(2) + D/2.
- 추출 원통 반경: 2R_D.
- z 범위: cases.csv의 ground_z부터 internalMesh의 최대 z까지 자동 추출.
- z는 OpenFOAM 실제 좌표이며 원점을 빼지 않습니다.
- 행은 지면에서 메시 상단으로 증가, 열은 theta=0~360도(끝점 제외).
- rotor_z는 물리 조건/메타데이터이며 추출 상한이 아닙니다.
- 로터면이 z=0이면 음수 z의 지면은 그림 아래, 양수 z의 메시 상단은 그림 위입니다.

입력 shape: (4, Nz, Ntheta), 기본 해상도 (256, 256)
- CH0: L/D
- CH1: sin(theta)
- CH2: cos(theta)
- CH3: z/R_D
- H/R_D 별도 상수 채널은 없습니다.

CSV: folder,L,D,rotor_z,ground_z,load,type,center_x,center_y
- L, D, rotor_z, ground_z, center_x/y 단위 m; load 단위 N/m².
- type=V는 검증 데이터로 학습에서 제외.
- center_x/y 생략 시 0. folder에는 Linux 경로를 사용.
- z_max를 CSV에 입력할 필요 없음. 추출에서 mesh.bounds[5]를 읽음.

출력
- inputs/*.npz: 실제 z_m, z_over_rd, z_max_m, 좌표 버전 및 케이스 메타데이터.
- targets_u/v/w/*.npy: 무차원 속도 배열. 행 순서는 z_m과 동일.
- Plot 세로축: z [m]. 예측 커서: theta, z, U/V/W, 속도 크기[m/s].
- 속도 W의 부호를 뒤집지 않음. 실제 z좌표에 맞는 행 배치만 사용.

실행: README_LINUX.md 참고
1. bash rotor.sh setup
2. bash rotor.sh extract --data-root /data/OpenFOAM
3. bash rotor.sh visualize
4. bash rotor.sh train --epochs 100
5. bash rotor.sh evaluate
6. bash rotor.sh predict --rotor-z 0 --ground-z -2 --z-max 3
마지막 명령의 좌표는 예시입니다. 실제 추출 로그에 나타난 메시 상단 값을 쓰세요.

이전 버전에서 전환
- 기존 s/R_D의 4채널과 H/R_D 포함 5채널 가중치 모두 사용할 수 없음.
- 새 범위로 재추출 및 재학습 필요. 기본 경로 dataset_zrd, models_zrd.
- coordinate_system=absolute_z_over_rd_v1로 데이터/가중치의 의미를 검사.
- 원통 전체가 유효한 CFD 영역에 들어가야 함. 영역 밖 샘플은 오류 처리.
- 이 4채널은 로터 위치/높이를 별도 조건으로 입력하지 않음. 다른 물리 조건을 무작정
  혼합해 학습하지 말고, 공통 원점과 일관된 로터 배치 조건을 사용하세요.
