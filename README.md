# Catch-Ping RecSys (TRI-AI)

> **Catch-Ping** 스마트 다이닝 플랫폼의 **AI 추천 엔진**입니다.
> 판교 지역 식당을 대상으로 개인화 추천 점수를 산출하며, 데이터 수집부터 모델 재학습까지
> 전 과정을 자동화한 MLOps 파이프라인과 함께 운영됩니다.
> 인프라·CI/CD·모니터링은 [TRI-Cloud](https://github.com/Trinity-goorm/TRI-Cloud) 레포가 담당합니다.

---

## 목차

1. [플랫폼 구성](#1-플랫폼-구성)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [추천 시스템](#3-추천-시스템)
4. [데이터 파이프라인 및 동기화](#4-데이터-파이프라인-및-동기화)
5. [MLOps](#5-mlops)
6. [API 엔드포인트](#6-api-엔드포인트)
7. [환경 변수](#7-환경-변수)
8. [로컬 실행](#8-로컬-실행)
9. [Docker 실행](#9-docker-실행)
10. [현황 및 한계](#10-현황-및-한계)
11. [데이터 및 라이선스](#11-데이터-및-라이선스)

---

## 1. 플랫폼 구성

| 레포지토리 | 역할 | 기술 스택 |
|---|---|---|
| [TRI-AI](https://github.com/Trinity-goorm/TRI-AI) (본 레포) | AI 추천 엔진 | FastAPI, scikit-learn, XGBoost, LightGBM, CatBoost, Pandas |
| [TRI-BE](https://github.com/Trinity-goorm/TRI-BE) | 백엔드 API | Spring Boot, Java, MySQL (AWS RDS), Redis, MongoDB |
| [TRI-FE](https://github.com/Trinity-goorm/TRI-FE) | 프론트엔드 | React, TanStack Query, Tailwind CSS |
| [TRI-Cloud](https://github.com/Trinity-goorm/TRI-Cloud) | 인프라·MLOps | AWS, Docker, GitHub Actions, Prometheus, Grafana |

플랫폼 핵심 기능: 6분 좌석 선점 후 결제, 예약 알림, 좌석 공석 알림, 12개 음식 카테고리 검색/필터, 개인화 추천.

---

## 2. 시스템 아키텍처

### 전체 데이터 흐름

```
[AWS RDS (MySQL)]
  restaurant, user, user_preference,
  user_preference_category, likes, reservation
        |
        | SQLAlchemy + pandas.read_sql
        | schedule.every(60min)  ← TRI-Cloud: rds_to_mongodb_sync.py
        | FULL REFRESH (delete_many → insert_many)
        v
[MongoDB]
  collections: restaurants, users, user_preferences,
               likes, reservations, recsys_data (per-user aggregate)
        |
        | pymongo fetch  ← TRI-AI: background_tasks.py
        | on startup (run_initial_sync)
        | + every 24h (periodic_data_sync, 초기 동기화 후 1h 대기)
        | → timestamped JSON → storage/input_json/ (최신 3개 유지)
        v
[TRI-AI: ML Pipeline]
  전처리 → 특성 엔지니어링 → 스태킹 앙상블 학습
        |
        v
[FastAPI 추천 서빙]  :5000
  POST /recommend  →  상위 15개 식당 반환
  추천 결과 → storage/output_feedback_json/
```

### 인프라 토폴로지 (TRI-Cloud 관리)

```
Internet
    |
[Bastion EC2]  (퍼블릭 서브넷)
    |  SSH ProxyJump
    +--[FE EC2]  (프라이빗 서브넷)
    +--[BE EC2]  (프라이빗 서브넷)
    +--[ML EC2]  (프라이빗 서브넷, port 5000)  ← TRI-AI 컨테이너
    +--[MongoDB EC2]  (프라이빗 서브넷, port 27017)
    |
[AWS RDS MySQL]  (ap-northeast-2)
[Monitoring EC2]  Prometheus:9090 / Grafana:3000 / node-exporter:9100
```

### CI/CD 흐름 (GitHub Actions)

```
git push → main 브랜치
    |
    | (TRI-AI: .github/workflows/deploy.yml)
    | (TRI-Cloud: github_action/ml/deploy.yml  ← 중앙화 관리)
    v
[GitHub Actions runner]
  1. configure-aws-credentials
  2. ECR login
  3. docker build + tag(${{ github.sha }}) + push → trinity-repo
  4. SSH ProxyJump (bastion → ML EC2)
  5. docker pull → stop/rm old → docker run -d -p 5000:5000
  6. Discord webhook 알림 (성공/실패)
```

---

## 3. 추천 시스템

### 모델 구조 — 스태킹 앙상블

```
입력 피처 (17개)
  review, duration_hours, conv_WIFI, conv_주차, caution_예약가능,
  log_review, review_duration,
  [특성 엔지니어링 파생] bayesian_rating, popularity_score,
  category_diversity_score, engagement_score, interaction_intensity,
  composite_rating, rating_vs_category, reviews_vs_category,
  category_quality_interaction, review_density
        |
        | StandardScaler → 80/20 분할
        v
┌─────────────────────────────────────────────┐
│  Base Learners (각각 GridSearch/RandomSearch, cv=3)  │
│   Ridge · RandomForest · XGBoost             │
│   LightGBM · CatBoost · MLP                 │
└──────────────┬──────────────────────────────┘
               | Out-of-fold predictions
               v
        Ridge Meta-Learner
        (StackingRegressor, cv=3)
               |
               v
        predicted_score
```

### 특성 엔지니어링 (enhance_feature_engineering)

| 파생 특성 | 설명 |
|---|---|
| `bayesian_rating` | `(review * score + 10 * global_avg) / (review + 10)` |
| `popularity_score` | `score * 0.6 + log1p(review) * 0.4` |
| `engagement_score` | `log1p(review) * 0.7 + (duration_hours/24) * 0.3` |
| `category_diversity_score` | 카테고리 내 희소성 (희소 카테고리일수록 높음) |
| `interaction_intensity` | `review * 0.4 + duration_hours * 0.25 + log1p(review) * 0.35` |
| `review_density` | `review / (duration_hours + 1)` |
| `composite_rating` | `score * 0.7 + diversity * 0.2 + rating_vs_category * 0.1` |
| `rating_vs_category` | 해당 카테고리 평균 대비 식당 평점 |
| `reviews_vs_category` | 해당 카테고리 평균 대비 리뷰 수 비율 |
| `category_quality_interaction` | `score * category_avg_rating` |
| `convenience_score` | `conv_*` 컬럼 합계 |

결측치 보완: `IterativeImputer(BayesianRidge)` — score·review 컬럼 대상.

### 복합 점수 (Composite Score)

```
composite_score
  = predicted_score (스태킹 모델 출력)
  + REVIEW_WEIGHT    * log(review + 50) / log(1000)
  + CAUTION_WEIGHT   * (긍정 유의사항 - 부정 유의사항)
  + CONVENIENCE_WEIGHT * mean(conv_* 컬럼)
  + category_bonus
  + category_diversity_bonus * 0.1

최종: sigmoid(composite_score, A_VALUE, B_VALUE)
    = 5 / (1 + exp(-A_VALUE * (x - B_VALUE)))
    → 0~5 범위로 변환
```

기본값: `REVIEW_WEIGHT=0.4`, `CAUTION_WEIGHT=0.15`, `CONVENIENCE_WEIGHT=0.15`,
`A_VALUE=1.25`, `B_VALUE=2.5`

### 기존 사용자 vs 신규 사용자 (Cold Start)

| 구분 | 판단 기준 | 추가 로직 |
|---|---|---|
| **기존 사용자** | `user_features_df`에 해당 `user_id` 존재 | `max_price` 필터링; 선호 카테고리 +0.3 보너스 (카테고리 4/7/9/10은 추가 +0.2); `completed_reservations > 3` → 예약가능 식당 +0.2; `like_to_reservation_ratio` 구간별 인기도 보너스 |
| **신규 사용자** | `user_features_df`에 없거나 데이터 없음 | 카테고리 일괄 +0.3; 다양성(희소 카테고리 +0.15); 인기도(log 스케일 +0.2); 운영시간 보너스(+0.1); 편의시설 보너스(+0.05/개) |

결과: 중복 제거(restaurant_id 기준) 후 상위 15개 반환.

### 추천 카테고리 매핑

| ID | 카테고리 | ID | 카테고리 | ID | 카테고리 |
|---|---|---|---|---|---|
| 1 | 중식 | 5 | 이탈리안 | 9 | 스테이크 |
| 2 | 일식집 | 6 | 이자카야 | 10 | 고깃집 |
| 3 | 브런치카페 | 7 | 한식집 | 11 | 다이닝바 |
| 4 | 파스타 | 8 | 치킨 | 12 | 오마카세 |

### 평가 지표 (app/services/evaluation/)

- 랭킹 지표: Precision@K, Recall@K, NDCG@K, Hit Rate@K (K = 5, 10, 15)
- 사용자 세그먼트별 성능: new (상호작용 0회) / active (> 10회) / inactive (그 외)
- 다양성 지표, K-Fold 교차 검증
- 하이퍼파라미터 최적화: `GET /evaluate/optimize-params` → Optuna 기반 (최대 50회 탐색, 다중 목표)
- 하이브리드 추천 (CF + 콘텐츠 기반): `app/services/model_trainer/recommenation/hybrid.py` 구현 — 평가 엔드포인트(`/evaluate/compare-algorithms`)에서만 사용 가능, 메인 `/recommend` 미적용

---

## 4. 데이터 파이프라인 및 동기화

두 단계가 **독립적인 주기**로 운영됩니다.

### Stage A — RDS → MongoDB (TRI-Cloud)

- 파일: `mongodb/mongodb_sync/rds_to_mongodb_sync.py`
- 실행: MongoDB EC2에서 Docker 컨테이너 또는 nohup 프로세스
- 주기: `schedule.every(SYNC_INTERVAL_MINUTES=60)`
- 방식: SQLAlchemy `create_engine` + `pandas.read_sql` → numpy 타입 변환 → **FULL REFRESH** (각 컬렉션 `delete_many({})` 후 `insert_many`)

| MongoDB 컬렉션 | RDS 원천 테이블 |
|---|---|
| `restaurants` | restaurant, restaurant_category |
| `users` | user |
| `user_preferences` | user_preference + user_preference_category (병합) |
| `likes` | likes |
| `reservations` | reservation |
| `recsys_data` | 위 전체를 사용자 단위로 집계한 통합 문서 |

### Stage B — MongoDB → 학습 및 서빙 (TRI-AI)

- 파일: `app/services/background_tasks.py`
- 기동 시: `run_initial_sync()` → `fetch_data_from_mongodb()` → JSON 저장 → `initialize_model(use_direct_mongodb=True)`
- 주기: `periodic_data_sync(hours_interval=SYNC_INTERVAL_HOURS=24)` — 초기 동기화 완료 후 **1시간 대기**, 이후 24시간 주기 반복
- JSON 파일: `storage/input_json/restaurants/`, `storage/input_json/user/` — 타입별 최신 3개 유지

```
[서버 기동]
  run_initial_sync()
    fetch_data_from_mongodb()  # MongoDB → timestamped JSON
    initialize_model(use_direct_mongodb=True)
      전처리 → 특성 엔지니어링 → 스태킹 앙상블 학습
  periodic_data_sync(24h) 백그라운드 시작
    (1h 대기)
    while True:
      fetch_data_from_mongodb()  # JSON 갱신
      initialize_model(force_reload=True, use_direct_mongodb=False)
      (24h - 소요시간) 대기
```

---

## 5. MLOps

### 컨테이너

- 베이스 이미지: `python:3.12-slim`
- LightGBM용 `libgomp1` 설치 (`apt-get install -y libgomp1`)
- `EXPOSE 5000`, `VOLUME ["/app/storage"]`
- ECR 레포지토리: `trinity-repo`, 이미지 태그: 커밋 SHA

Dockerfile 위치: TRI-AI `Dockerfile`, TRI-Cloud `dockerfile/AI/Dockerfile` (중앙화 관리용).

### CI/CD — GitHub Actions

트리거: `main` 브랜치 push

필요한 GitHub Secrets:

| Secret | 설명 |
|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` | ECR 인증 |
| `DEPLOY_SSH_KEY` | Bastion 및 ML EC2 SSH 키 |
| `MONGO_HOST` / `MONGO_PORT` / `MONGO_USER` / `MONGO_PASSWORD` / `MONGO_DATABASE` / `MONGO_COLLECTION` | MongoDB 접속 정보 |
| `DISCORD_WEBHOOK` | 배포 결과 알림 |

### 모니터링 (TRI-Cloud: monitoring/)

스택: `monitoring/docker-compose.yml`로 Monitoring EC2에서 운영.

| 컴포넌트 | 포트 | 역할 |
|---|---|---|
| Prometheus | 9090 | 메트릭 수집, 15일 보존 |
| Grafana | 3000 | 대시보드 (`/monitoring` 서브패스 서빙) |
| node-exporter | 9100 | 호스트 OS 메트릭 |
| cloudwatch-exporter | 9106 | RDS CloudWatch 메트릭 (CPU, 연결수, IOPS, 레이턴시 등) |

`prometheus.yml` 스크레이프 대상 (15s 간격):

| job | 대상 |
|---|---|
| `node_exporter` | Monitoring EC2 |
| `frontend` | FE EC2 (`<private-subnet-IP>:9100`) |
| `backend` | BE EC2 (`<private-subnet-IP>:9100`) |
| `ml-server` | ML EC2 (`<private-subnet-IP>:9100`) |
| `mongodb_node` | MongoDB EC2 (`<private-subnet-IP>:9100`) |
| `mongodb_db` | MongoDB 익스포터 (`<private-subnet-IP>:9216`) |
| `cloudwatch` | cloudwatch-exporter (60s 간격) |

---

## 6. API 엔드포인트

추천 라우터는 `/recommend` prefix 하위에 마운트됩니다.

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/` | 서버 상태 및 docs URL |
| `POST` | `/recommend` | 개인화 추천 생성 (메인 엔드포인트) |
| `POST` | `/recommend/reload` | 모델 강제 재초기화 (관리자용) |
| `GET` | `/recommend/status` | 모델 초기화 상태 및 통계 조회 |
| `GET` | `/recommend/evaluate` | 추천 모델 성능 지표 계산 |
| `GET` | `/docs` | Swagger UI |

평가 전용 라우터(`app/router/evaluation.py`)에는 `/evaluate/basic`, `/evaluate/cross-validation`, `/evaluate/optimize-params`, `/evaluate/compare-algorithms`가 정의되어 있으나, 현재 `main.py`에는 등록되어 있지 않습니다. 필요 시 `app.include_router`로 추가 등록할 수 있습니다.

### 추천 요청 예시

```http
POST /recommend
Content-Type: application/json
```

```json
{
  "userId": 1024,
  "preferredCategories": ["한식집", "고깃집", "파스타"]
}
```

- `userId`: 사용자 ID (정수)
- `preferredCategories`: 선호 카테고리 이름 배열 (1~3개)

### 추천 응답 예시

```json
{
  "user": 1024,
  "is_new_user": false,
  "recommendations": [
    {
      "category_id": 7,
      "restaurant_id": 215,
      "score": 4.6,
      "predicted_score": 4.512,
      "composite_score": 4.873
    }
  ]
}
```

- `score`: 원본 식당 평점
- `predicted_score`: 스태킹 앙상블 예측값
- `composite_score`: 시그모이드 변환 후 최종 점수 (0~5)

추천 결과는 백그라운드 작업으로 `storage/output_feedback_json/recommendation_{userId}_{timestamp}.json`에도 저장됩니다.

모델 초기화 완료 전 요청 시 `503 Service Unavailable` (`Retry-After` 헤더 포함)이 반환됩니다.

---

## 7. 환경 변수

`.env` 파일 또는 컨테이너 환경 변수로 설정합니다.

### MongoDB 접속

| 변수 | 설명 | 기본값 |
|---|---|---|
| `MONGO_HOST` | MongoDB 호스트 | (필수) |
| `MONGO_PORT` | 포트 | `27017` |
| `MONGO_USER` | 사용자명 | (필수) |
| `MONGO_PASSWORD` | 비밀번호 | (필수) |
| `MONGO_DATABASE` | 데이터베이스명 | (필수) |
| `MONGO_COLLECTION` | 추천용 집계 컬렉션 | `recsys_data` |

### SSH 터널링 / 동기화

| 변수 | 설명 | 기본값 |
|---|---|---|
| `USE_SSH_TUNNEL` | SSH 터널 사용 여부 | `false` |
| `SSH_HOST`, `SSH_PORT`, `SSH_USER`, `SSH_PASSWORD`, `SSH_KEY_PATH` | 터널 설정 | — |
| `SYNC_INTERVAL_HOURS` | 주기적 재학습 간격(시간), `0`이면 비활성 | `24` |

### 저장 경로

| 변수 | 설명 | 기본값 |
|---|---|---|
| `STORAGE_DIR` | 저장소 루트 | `storage` |
| `RESTAURANTS_DIR` | 식당 JSON 경로 | `storage/input_json/restaurants` |
| `USER_DIR` | 사용자 JSON 경로 | `storage/input_json/user` |
| `FEEDBACK_DIR` | 추천 결과 경로 | `storage/output_feedback_json` |

### 추천 점수 가중치 (app/setting.py)

| 변수 | 설명 | 기본값 |
|---|---|---|
| `A_VALUE` | 시그모이드 기울기 | `1.25` |
| `B_VALUE` | 시그모이드 중심점 | `2.5` |
| `REVIEW_WEIGHT` | 리뷰 가중치 | `0.4` |
| `CAUTION_WEIGHT` | 유의사항 가중치 | `0.15` |
| `CONVENIENCE_WEIGHT` | 편의시설 가중치 | `0.15` |

TRI-Cloud `rds_to_mongodb_sync.py` 쪽 환경 변수:

| 변수 | 설명 | 기본값 |
|---|---|---|
| `RDS_HOST` | RDS 엔드포인트 | (필수, 예: `<RDS_ENDPOINT>`) |
| `RDS_USER` / `RDS_PASSWORD` / `RDS_DATABASE` / `RDS_PORT` | RDS 접속 정보 | (필수) |
| `SYNC_INTERVAL_MINUTES` | RDS→Mongo 동기화 간격 | `60` |

---

## 8. 로컬 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

LightGBM 사용을 위해 OpenMP 런타임이 필요합니다.

- Debian/Ubuntu: `apt-get install libgomp1`
- macOS: `brew install libomp`

### 2. 환경 변수 설정

```bash
# .env 파일 생성 후 아래 항목을 채웁니다.
MONGO_HOST=<MONGO_HOST>
MONGO_PORT=27017
MONGO_USER=<MONGO_USER>
MONGO_PASSWORD=<MONGO_PASSWORD>
MONGO_DATABASE=<MONGO_DATABASE>
MONGO_COLLECTION=recsys_data
USE_SSH_TUNNEL=false
```

### 3. 서버 실행

```bash
python main.py
# 또는
uvicorn main:app --host 0.0.0.0 --port 5000 --reload --log-config logging_config.json
```

- 기본 포트: **5000**
- API 문서: http://localhost:5000/docs

서버 기동 시 MongoDB 동기화 → 모델 초기화가 자동으로 수행됩니다.

---

## 9. Docker 실행

```bash
docker build -t catch-ping-recsys .

docker run -d --name trinity-container \
  -p 5000:5000 \
  -v $(pwd)/storage:/app/storage \
  -e MONGO_HOST=<MONGO_HOST> \
  -e MONGO_PORT=27017 \
  -e MONGO_USER=<MONGO_USER> \
  -e MONGO_PASSWORD=<MONGO_PASSWORD> \
  -e MONGO_DATABASE=<MONGO_DATABASE> \
  -e MONGO_COLLECTION=recsys_data \
  -e USE_SSH_TUNNEL=false \
  catch-ping-recsys
```

- 베이스 이미지: `python:3.12-slim`
- `/app/storage` 볼륨 마운트로 JSON 캐시 및 추천 결과를 컨테이너 외부에 보존합니다.

---

## 10. 현황 및 한계

아래는 소스 코드에서 직접 확인한 미완성 사항입니다.

| 항목 | 내용 |
|---|---|
| **Optuna 파라미터 자동 적용 불가** | `/evaluate/optimize-params`가 최적 파라미터를 반환하지만, 가중치(`A_VALUE`, `B_VALUE` 등)는 서버 기동 시 환경 변수로 읽힌다. 최적값을 반영하려면 환경 변수를 갱신하고 재배포해야 한다. |
| **하이브리드 추천 미연동** | `hybrid.py`(CF + 콘텐츠 기반)는 구현되어 있으나 `/evaluate/compare-algorithms` 평가 전용이며, 메인 `/recommend`에는 적용되어 있지 않다. |
| **MAE/RMSE 항상 None** | `calculate_rating_metrics`는 구현되어 있으나, 평가 파이프라인에서 `y_true` 기반 실측 평점이 공급되지 않아 실제 동작하지 않는다. |
| **Alertmanager 미구성** | `prometheus.yml`에 `alertmanagers` 항목이 주석 처리되어 있고, `rule_files`도 비어 있다. 알림 규칙이 없으므로 이상 감지 알림은 작동하지 않는다. |
| **storage 볼륨 마운트 누락** | TRI-Cloud `github_action/ml/deploy.yml`의 `docker run` 명령에 `-v` 마운트가 없다. 배포마다 컨테이너가 교체되면 `storage/` 내 JSON 캐시와 피드백 파일이 초기화된다. |
| **헬스체크 없이 Discord 성공 알림** | 워크플로가 `docker run` 직후 Discord에 성공 알림을 보낸다. 컨테이너가 기동은 됐지만 모델 초기화에 실패한 경우에도 성공으로 표시된다. |
| **RDS→Mongo FULL REFRESH 위험** | 동기화는 전체 삭제 후 재삽입 방식이다. RDS 장애가 동기화 도중 발생하면 컬렉션이 빈 상태로 남아 다음 성공 사이클까지 추천이 불가능하다. |

---

## 11. 데이터 및 라이선스

### 데이터

- `data/crawling_1st_data/`, `data/crawling_2nd_data/`: 카카오맵 판교 지역 식당·메뉴 크롤링 원본 (CSV/JSON)
- `data/example_data/`: 사용자·예약·선호도 예시 테이블
- `data/Trinity_Selenium_crawling(식당정보 & 메뉴 크롤러).ipynb`: Selenium 크롤러
- `research & test/`: 추천 시스템 프로토타입 노트북 (v1.0, v1.1)

### 라이선스

[LICENSE](./LICENSE) 파일을 참고하세요.
