# AGENTS.md

Technical documentation for AI agents working on this codebase. User-facing
docs (what it does, how to install, how to run) live in `README.md`.

## Repository Layout

- Python training core (no package name; scripts run from the repo root):
  - `run.py` - CLI entrypoint; loads a job config and runs the job.
  - `toolkit/` - model wrappers, configs, metadata, `job.py` (job state
    writes: status/step/queue updates over the shared SQLite DB).
  - `jobs/process/BaseSDTrainProcess.py` - the LoRA/full training loop:
    checkpoint saving, resume, optimizer state, critic (GAN) handling.
  - `extensions_built_in/sd_trainer/DiffusionTrainer.py` - `diffusion_trainer`
    process: launches `BaseSDTrainProcess`, reports status back to the UI.
  - `models/critic.py` - GAN critic state save/resume.
  - `manager/` - the AI Toolkit Manager (hardware detection, install, FFmpeg).
- `ui/` - Next.js 15 (App Router) web UI, TypeScript:
  - `ui/src/app/api/` - route handlers (jobs, queues, files, settings, ...).
  - `ui/src/server/` - server-side modules shared by routes (prisma
    singleton, settings cache, forkJob, monitor).
  - `ui/src/components/`, `ui/src/utils/` - React components and client
    API helpers (`utils/jobs.ts` wraps axios calls).
  - `ui/cron/` - the UI's Node worker processes (compiled separately via
    `tsconfig.worker.json`): `worker.ts` polls the job queue every second,
    `startJob.ts` launches `python run.py <.job_config.json>` with the
    `AITK_JOB_ID` env var, `fileServer.ts` is a multi-process file
    server + reverse proxy that supervises the Next.js server.
  - `ui/prisma/schema.prisma` - SQLite schema (`aitk_db.db` at the repo
    root). Models: `Job`, `Queue`, `Settings`.
- `config/examples/` - example training configs; `output/` - job output
  folders (one folder per job name); `datasets/` - training data;
  `aitk_db.db` - the shared SQLite database (WAL).

## Job System (how a job lives)

1. `POST /api/jobs` creates a `Job` row (`name` @unique, `job_config` is a
   JSON string, `status` defaults to `stopped`).
2. `GET /api/jobs/[id]/start` sets `status='queued'` + a queue position and
   ensures a `Queue` row exists for the job's `gpu_ids`.
3. The Node worker (`ui/cron/worker.ts`, 1s poll) picks queued jobs whose
   `Queue` row allows it and spawns `python run.py` on a per-job temp copy
   of the config (`<output>/<job name>/.job_config.json`).
4. The Python side reports status/step/queue state back into `aitk_db.db`
   (`toolkit/job.py`); on stop, `Job.step` ends up equal to the last saved
   checkpoint step.
5. `GET /api/jobs/[id]/delete` removes the output folder and the row.

`Job.status` values in use: `stopped`, `queued`, `running`, `stopping`,
`completed`, `error`. `Job.stop` is a separate boolean (set by the stop
request; the process checks it between steps).

## Checkpoints and Resume (invariants - do not break)

All in the job's output folder (`<training root>/<job name>/`, training
root = the `TRAINING_FOLDER` setting, default `output/`):

- The trainer saves a bare `<name>.safetensors` on every save plus a
  step-suffixed `<name>_<9-digit>.safetensors`. With `save.save_format:
  diffusers`, the checkpoint is a directory instead (`<name>/`,
  `<name>_<9-digit>/`). `clean_up_saves()` prunes to
  `save.max_step_saves_to_keep`.
- Resume: `BaseSDTrainProcess.get_latest_save_path()` picks the newest
  (by `os.path.getctime`) item in the folder starting with the job name
  (`.safetensors` file, `.pt` file, or directory), skipping
  `_LoRA`/`_refiner`/`_t2i`/`_cn` suffixed items unless the job name itself
  contains that marker. `load_training_state_from_metadata()` reads the
  safetensors `training_info` metadata (a JSON string:
  `{"step": N, "epoch": E}`) to restore step/epoch - only when
  `train.start_step` is not set. The optimizer is loaded from
  `optimizer.pt` in the same folder (skipped if the network weights were
  changed at load time, `did_change_weights`).
- GAN critic state: newest `CRITIC_<name>*.safetensors`, restored via
  max mtime (`models/critic.py`).
- `optimizer.pt`, `learnable_snr.json` and `CRITIC_*` files are never
  treated as weight checkpoints by the resume scan.
- Consequence: a job's entire resumable state is "files in its own folder
  whose names start with the job name" + `optimizer.pt`. Anything that
  replicates a job for continued training must preserve this naming.

## Fork Job feature

Forking a training job creates a new job that resumes from the original's
current checkpoint while the original continues unchanged.

- Implementation is Node-side only (`ui/src/server/forkJob.ts` +
  `POST /api/jobs/[jobID]/fork`): copy the latest checkpoint (renamed to
  the fork's name so the trainer's `{name}*` resume scan in the fork's own
  folder finds it), plus `optimizer.pt`, `learnable_snr.json` (when
  present) and the newest `CRITIC_<name>*.safetensors` (when present),
  then insert a new `Job` row with `config.name`/`meta.name` rewritten to
  the fork name and `step`/`total_steps`/`gpu_ids` copied. No Python
  changes: auto-resume does the rest.
- Fork name: `<name>_fork`, then `<name>_fork2`, `<name>_fork3`, ...
  (`Job.name` is @unique; the output folder is derived from the name).
- Only forkable when the job is not `running`/`stopping` (HTTP 409
  otherwise) - `torch.save` of `optimizer.pt` is not atomic, so a
  mid-write copy could be torn. Only `train` jobs (400 otherwise); 400
  when the job has no checkpoint yet; 404 when the output folder is gone.
- New job starts `stopped` (shown as paused); the user edits settings,
  then starts it. `train.start_step` must stay unset in the config so
  auto-detect from metadata applies.
- Not restored (pre-existing resume limitation, applies to normal resume
  too): EMA shadow params, RNG state, dataloader position.
- The UI entry point is the "Fork Job" item in the job gear menu
  (`ui/src/components/JobActionBar.tsx`), enabled via `canFork` in
  `getAvaliableJobActions` (`ui/src/utils/jobs.ts`).

## Conventions and Gotchas

- One shared PrismaClient for the Next.js server: import
  `@/server/prisma`; never construct new clients in route handlers.
- Route handlers: Next 15 App Router; `params` is awaited
  (`const { jobID } = await params;`).
- The UI is built with `npm` (`ui/package-lock.json` is the lockfile; do
  not migrate it to another package manager). `npm run build` =
  `tsc -p tsconfig.worker.json && next build`.
- Known pre-existing issue: `ui/.next/types` contains validators generated
  by a newer Next.js than the installed one, so a bare `tsc --noEmit` in
  `ui/` reports `TS2344` errors for every route's `params` signature.
  These are stale-generated artifacts, not source errors; type-check the
  real source with a tsconfig that excludes `.next`.
- The `fileServer.ts` supervisor spawns Next.js on an ephemeral port and
  proxies; file/thumbnail requests (`/api/files/`, `/api/img/`,
  `/api/audio/art/`) are served by the file server directly.
- Job configs in the DB predate the current `JobConfig` type; parse as the
  domain type but guard fields you touch.
- Python: 3.10-3.12; heavy imports (torch) make the Python side slow to
  start - don't import `toolkit` from long-running Node tooling.
