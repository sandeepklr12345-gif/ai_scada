# AI_SCADA

AI_SCADA is a research and integration project for power output forecasting and industrial control system (ICS) anomaly analysis. It includes a 600 MW unit forecasting path, HAI 23.05 security and anomaly analysis, a FastAPI service, MQTT replay integration, PostgreSQL persistence, decision support logic, and a React plant dashboard.

The current runtime validation uses historical data replay. This repository does not establish a connection to a live power plant or authorize control of plant equipment.

## Current architecture

The validated 600 MW replay path is:

```text
Historical 600 MW replay CSV
  -> scripts/mqtt/scada_replay_publisher.py
  -> MQTT topic ai_scada/scada/600mw
  -> scripts/mqtt/scada_ai_subscriber.py
  -> ReplayRuntimeFeatureAdapter600MW
  -> RuntimeFeatureBuilder600MW
  -> 119 runtime generated features in frozen manifest order
  -> POST /forecast on FastAPI
  -> existing frozen Forecasting600MW models
  -> forecast response to the MQTT subscriber
  -> PostgreSQL measurements and ai_predictions writes
```

The adapter reads the replay timestamp and the 71 exact raw source fields. It does not use the replay CSV's already engineered lag or time fields as builder inputs. The FastAPI `/forecast` route receives only the generated 119-feature dictionary. The MQTT subscriber writes the original power measurement and the three forecast outputs after a successful forecast response.

The broader API also exposes HAI anomaly analysis, decision support, and latest-record PostgreSQL reads. The React frontend polls `/health` and `/models` for service and model status; it does not call `/runtime/latest` or `/predictions/latest`. Those API reads return stored records and do not establish a live plant feed. The frontend does not receive MQTT status, database status, alert history, or decision history, and its plant view and panels show unavailable or empty values.

## 600 MW power forecasting

The source workbook is `data/raw/600mw/600 MW unit one-week operating data.xlsx`. The three frozen LinearRegression models forecast power output at +2, +10, and +30 minutes. Runtime does not retrain these models.

The frozen model expects exactly 119 features:

- 71 `RAW_SOURCE` variables from the 600 MW source data
- 44 `LAG_DERIVED` variables, generated at row lag steps 1, 2, 5, and 10 from 11 source variables
- 4 `TIME_DERIVED` variables: hour, minute, day of week, and day of month

The feature names and their order are defined by `models/forecasting/600mw/600mw_forecasting_feature_manifest.json`. Model artifacts and metadata are in `models/forecasting/600mw/`.

### Historical validation results

These metrics describe offline temporal and replay validation. They are not live plant performance measurements.

| Validation | Forecast horizon | R² | RMSE |
|---|---:|---:|---:|
| Temporal robustness | +2 min | 0.998253 | 3.639 MW |
| Temporal robustness | +10 min | 0.974143 | 12.599 MW |
| Temporal robustness | +30 min | 0.814217 | 34.491 MW |
| Historical replay | +2 min | 0.9993 | 2.34 MW |
| Historical replay | +10 min | 0.9893 | 9.28 MW |
| Historical replay | +30 min | 0.9413 | 21.75 MW |

## Runtime feature construction

The runtime pipeline is implemented in:

- `scripts/integration/runtime_feature_builder_600mw.py`
- `scripts/integration/replay_runtime_feature_adapter_600mw.py`
- `scripts/integration/validate_600mw_runtime_source_contract.py`

The builder validates the required source names, timestamp ordering, sampling cadence, generated feature set, and exact model manifest order. Lag values use chronological row shifts, matching model training. Rows are withheld when there is not enough history or when the required lag window has an irregular interval. The replay adapter keeps the raw source history in memory and passes only timestamped raw values to the builder.

In the validated MQTT run, messages 1 through 10 were `NOT_READY` while the lag history filled. Message 11 was the first ready row. Each ready row contained exactly 119 finite values in manifest order. The source contract records the required 71 raw fields and the expected two minute sampling interval.

## MQTT replay integration

`scripts/mqtt/scada_replay_publisher.py` publishes the historical replay file `data/integration/runtime_simulation/600mw_scada_replay.csv` to `ai_scada/scada/600mw`. The replay contains 5,649 rows and the 71 raw source variables. It is a replay and test source, not a real plant SCADA feed.

The controlled production subscriber validation processed 20 MQTT messages:

- 10 warm-up messages and 10 ready messages
- 10 successful forecast requests and no forecast failures
- 10 PostgreSQL measurement writes and 30 prediction writes
- no invalid messages, ordering failures, or cadence failures
- exact 119-feature model schema and no NaN or infinite values

This is a bounded replay validation result. It does not demonstrate continuous service operation or live plant connectivity. A later attempt to repeat the combined validation was blocked before replay by missing PostgreSQL credentials and performed no writes; that attempt is not reported as a pass.

The relevant validation scripts include:

- `scripts/integration/test_replay_runtime_feature_adapter_600mw.py`
- `scripts/mqtt/test_600mw_runtime_feature_adapter_mqtt.py`
- `scripts/mqtt/test_600mw_runtime_forecasting_mqtt.py`
- `scripts/mqtt/test_600mw_runtime_fastapi_mqtt.py`

These checks cover runtime feature construction, MQTT-shaped messages, direct frozen-model inference, the FastAPI forecast interface, and controlled MQTT replay to PostgreSQL persistence. The source contract is checked by `scripts/integration/validate_600mw_runtime_source_contract.py`.

## FastAPI service

The application is `scripts/api/main.py`. It loads the 600 MW forecast models, HAI Candidate C anomaly engine, HAI attack-classifier components, decision engine, and PostgreSQL storage helper.

| Method | Endpoint | Implemented behavior |
|---|---|---|
| GET | `/health` | Reports API service status. |
| GET | `/models` | Reports model names, horizons, feature counts, and supported domains. |
| POST | `/forecast` | Accepts `{"features": {name: number}}` and returns `power_2min`, `power_10min`, and `power_30min`. |
| POST | `/anomaly` | Runs the HAI 23.05 Candidate C streaming anomaly inference. |
| POST | `/decision` | Evaluates a supported 600 MW or HAI feature schema and returns a decision level and action. |
| GET | `/runtime/latest` | Reads the latest configured measurement from PostgreSQL. |
| GET | `/predictions/latest` | Reads the latest three configured forecast predictions from PostgreSQL. |

The `/forecast` handler performs model inference in process and does not access PostgreSQL. PostgreSQL is used by the MQTT subscriber for persistence and by the latest-record API endpoints for reads.

From the repository root, a local API process can be started with:

```powershell
python -m uvicorn scripts.api.main:app --host 0.0.0.0 --port 8000
```

The repository's `start_api.ps1` starts the same app module through its configured WSL environment.

## PostgreSQL integration

The configured database name is `ai_scada`. The current integration uses these reference tables: `plants`, `data_sources`, `equipment`, and `parameters`. Runtime data is stored in `measurements` and `ai_predictions`.

The 600 MW MQTT subscriber persists the replay's raw power output as a measurement and the successful +2, +10, and +30 minute forecasts as prediction rows. PostgreSQL currently contains replay or reference integration data; it is not a source of live plant telemetry. The validated 600 MW runtime feature input comes from the historical replay, not from a complete set of PostgreSQL SCADA tags.

Database settings use these environment variables, matching `scripts/integration/postgres_scada_reader.py` and the PostgreSQL storage helper:

- `AI_SCADA_DB_HOST` (default `localhost`)
- `AI_SCADA_DB_PORT` (default `5432`)
- `AI_SCADA_DB_NAME` (default `ai_scada`)
- `AI_SCADA_DB_USER` (default `postgres`)
- `AI_SCADA_DB_PASSWORD` (required when PostgreSQL authentication requires it)

### Windows PowerShell

```powershell
$env:AI_SCADA_DB_HOST="localhost"
$env:AI_SCADA_DB_PORT="5432"
$env:AI_SCADA_DB_NAME="ai_scada"
$env:AI_SCADA_DB_USER="postgres"
$env:AI_SCADA_DB_PASSWORD="<your-password>"
```

### WSL

When PostgreSQL runs on Windows, WSL may need to use the Windows gateway address instead of `localhost`. Check the gateway with `ip route show default`, then set it as the host:

```bash
export AI_SCADA_DB_HOST="$(ip route show default | awk '{print $3}')"
export AI_SCADA_DB_PORT="5432"
export AI_SCADA_DB_NAME="ai_scada"
export AI_SCADA_DB_USER="postgres"
export AI_SCADA_DB_PASSWORD="<your-password>"
```

The root `.gitignore` excludes `.env` files. The Python code does not automatically load a `.env` file; export the variables in the shell or configure them in the process manager. Do not commit database passwords.

## Decision support

`scripts/decision_support/decision_engine.py` returns decision levels and actions based on the supplied anomaly state and persistence count:

- `NORMAL` → Level 1 (`AUTOMATIC` action label)
- `ANOMALY` → Level 2 (`APPROVAL_REQUIRED`)
- persistent anomaly count of 3 or more → Level 3 (`HUMAN_ONLY`)
- `INSUFFICIENT_HISTORY` or `UNAVAILABLE` → Level 2 (`APPROVAL_REQUIRED`)

Level 1 is an API decision result. No code path here sends control commands to plant equipment. For 600 MW features, the API reports anomaly assessment as unavailable because no compatible 600 MW anomaly detector is implemented. HAI Candidate C is used only with the HAI input schema.

## HAI 23.05 security and anomaly analysis

The HAI 23.05 work evaluates ICS anomaly and attack behavior in the HAI dataset. It is not a validated detector of physical plant faults. The frozen Candidate C Isolation Forest uses 118 selected features derived by the HAI inference pipeline. Its reported validation F1 scores are approximately 0.235 on Test 1 and 0.197 on Test 2. These are dataset evaluation results, not plant deployment metrics.

The repository also contains attack-label, temporal classifier, calibration, scenario-mapping, and decision-support work. The experimental temporal attack classifier uses 408 features for 39 mechanisms. In the evaluated fixed 0.5 threshold setup, F1 was zero; other threshold sweeps produced only low scores. Its outputs are not validated as reliable operational probabilities or an operational attack classifier. These HAI research results should not be presented as a validated plant-fault detector.

## React frontend

The `frontend/` directory contains the React dashboard and interactive plant view. The dashboard checks API health and model information. It does not call the API's latest measurement or prediction endpoints, and it shows that live measurements, MQTT status, database status, alert history, and live time series are unavailable. Parameter and chart panels remain placeholders until an appropriate data feed is connected.

## Validation and testing

The repository's validation work covers:

- standalone runtime feature-builder and replay-adapter validation
- MQTT feature-adapter validation
- MQTT-to-frozen-model inference validation
- MQTT-to-FastAPI forecast validation
- controlled MQTT replay to PostgreSQL persistence validation
- 600 MW runtime source-contract validation
- HAI Candidate C schema and inference validation

The 600 MW MQTT checks use a local broker and the historical publisher. Start the broker if needed, start the selected subscriber or validation script, and run the replay publisher separately. The subscriber tests wait for MQTT messages; they do not launch the publisher. Database-backed validation additionally requires the PostgreSQL environment variables above.

## Repository structure

```text
config/                              Project paths and forecasting configuration
data/
  raw/                               600 MW, HAI, Steam Generator, and UCI CCPP source data
  features/                          Model feature data and HAI Candidate C artifacts
  integration/                       Runtime contracts, replay input, validation reports
models/
  forecasting/600mw/                 Frozen forecast models and feature manifest
  attack_classification/hai_2305/    HAI attack-classifier artifacts and evaluations
frontend/                            React dashboard
scripts/
  api/                               FastAPI application
  database/                          PostgreSQL storage and database checks
  decision_support/                  Decision engine and HAI analysis
  inference/                          Forecast and anomaly inference code
  integration/                        Runtime adapter, feature builder, and contracts
  mqtt/                              Replay publisher and MQTT subscribers/tests
  cleaning/                           Data cleaning
  dataset_creation/                   Dataset and target construction
  inspection/                         HAI feature and evaluation workflows
  modeling/                            Forecast model validation and freezing
  processing/                          Dataset and artifact utilities
  validation/                          Validation report generation
README.md
requirements.txt
```

The root `.gitignore` excludes generated content under `data/` while allowing `data/raw/**`. A clean checkout may need the required local feature data and model artifacts before the API and inference workflows can load them.

## Setup

The workspace has been used with Python 3.11; the repository does not declare a formal Python version constraint.

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements cover the implemented Python API, frozen model inference, data preparation and validation, PostgreSQL access, and MQTT replay integration. The MQTT subscriber uses the Paho MQTT 2.x callback API. CUDA, PyTorch, CuPy, and cuDF are not required by the documented runtime path; GPU experiments are separate from the supported runtime dependencies.

## Limitations and current status

- The MQTT publisher is a historical replay/test source, not live plant SCADA.
- PostgreSQL holds replay or reference integration data; this does not demonstrate live SCADA ingestion or a complete 71-variable PostgreSQL source.
- The 600 MW forecasting models are frozen and infer from exactly 119 runtime-built features; runtime retraining is not implemented.
- HAI Candidate C and attack-classifier results are research and dataset validation. They are not a validated plant-fault detector or an operationally validated attack-probability service.
- Decision support returns a level and action label. It does not autonomously control plant equipment.
- The React frontend has no live measurement, MQTT, database, alert-history, or decision-history feed from the current API.
