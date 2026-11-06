import fs from 'fs';
import path from 'path';
import { Job } from '@prisma/client';
import { JobConfig } from '@/types';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';

// Error carrying the HTTP status the route should answer with.
export class ForkJobError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export interface ForkJobResult {
  name: string;
  jobConfig: JobConfig;
  jobConfigJson: string;
  step: number;
  totalSteps: number | null;
  // What was copied into the fork's output folder, for logging/UI messages.
  copied: string[];
}

// Job states in which it is safe to copy checkpoint files off disk. While a
// job is running (or stopping) the trainer may be mid-write on the
// checkpoint and optimizer files, so a copy could be torn.
const FORKABLE_STATUSES = ['stopped', 'queued', 'completed', 'error'];

// Mirror of BaseSDTrainProcess.get_latest_save_path: the trainer resumes from
// the newest item in its save folder that starts with the job name
// (safetensors file, .pt file, or a diffusers-format directory), skipping the
// auxiliary save families (_LoRA, _refiner, _t2i, _cn) unless the job name
// itself belongs to that family. Keeping the two in sync is what makes a fork
// resume from exactly the state the original job saved.
interface CheckpointCandidate {
  path: string;
  ctimeMs: number;
  isDir: boolean;
}

const findCheckpointCandidates = (srcFolder: string, jobName: string): CheckpointCandidate[] => {
  const candidates: CheckpointCandidate[] = [];
  for (const entry of fs.readdirSync(srcFolder)) {
    if (!entry.startsWith(jobName)) continue;
    if (entry === 'optimizer.pt') continue; // optimizer state is copied separately
    if (!jobName.includes('_LoRA') && entry.includes('_LoRA')) continue;
    if (!jobName.includes('_refiner') && entry.includes('_refiner')) continue;
    if (!jobName.includes('_t2i') && entry.includes('_t2i')) continue;
    if (!jobName.includes('_cn') && entry.includes('_cn')) continue;

    let stats: fs.Stats;
    try {
      stats = fs.statSync(path.join(srcFolder, entry));
    } catch {
      continue;
    }
    if (stats.isDirectory()) {
      // diffusers save format writes a folder; a folder literally named like a
      // file is not a checkpoint
      if (entry.endsWith('.safetensors')) continue;
      candidates.push({ path: path.join(srcFolder, entry), ctimeMs: stats.ctimeMs, isDir: true });
    } else if (entry.endsWith('.safetensors') || entry.endsWith('.pt')) {
      candidates.push({ path: path.join(srcFolder, entry), ctimeMs: stats.ctimeMs, isDir: false });
    }
  }
  return candidates.sort((a, b) => a.ctimeMs - b.ctimeMs);
};

// The latest checkpoint in the source folder, or null if the job has not
// saved one yet.
const findLatestCheckpoint = (srcFolder: string, jobName: string): CheckpointCandidate | null => {
  const candidates = findCheckpointCandidates(srcFolder, jobName);
  return candidates.length > 0 ? candidates[candidates.length - 1] : null;
};

// Pick the next free fork name: <name>_fork, then <name>_fork2, <name>_fork3, ...
const pickForkName = async (jobName: string, isNameTaken: (name: string) => Promise<boolean>): Promise<string> => {
  const base = `${jobName}_fork`;
  for (let i = 1; ; i++) {
    const candidate = i === 1 ? base : `${base}${i}`;
    if (!(await isNameTaken(candidate))) return candidate;
  }
};

// Copy the training state a fork needs to resume: the latest checkpoint
// (renamed to the fork's name so the trainer's resume scan finds it in its
// own save folder), the optimizer state, the learnable-SNR state and the
// latest GAN critic checkpoint when present.
const copyForkState = (
  srcFolder: string,
  forkFolder: string,
  jobName: string,
  forkName: string,
  checkpoint: CheckpointCandidate,
): string[] => {
  const copied: string[] = [];

  if (checkpoint.isDir) {
    const dest = path.join(forkFolder, forkName);
    fs.cpSync(checkpoint.path, dest, { recursive: true });
    copied.push(`${path.basename(checkpoint.path)}/ -> ${forkName}/`);
  } else {
    const ext = path.extname(checkpoint.path) || '.safetensors';
    const dest = path.join(forkFolder, `${forkName}${ext}`);
    fs.copyFileSync(checkpoint.path, dest);
    copied.push(`${path.basename(checkpoint.path)} -> ${forkName}${ext}`);
  }

  for (const file of ['optimizer.pt', 'learnable_snr.json']) {
    const src = path.join(srcFolder, file);
    try {
      fs.statSync(src);
    } catch {
      continue;
    }
    fs.copyFileSync(src, path.join(forkFolder, file));
    copied.push(file);
  }

  // GAN critic state (flex/critic training): the trainer resumes the critic
  // from the newest CRITIC_<job name>*.safetensors in its save folder.
  let critic: string | null = null;
  let criticCtime = -1;
  for (const entry of fs.readdirSync(srcFolder)) {
    if (!entry.startsWith(`CRITIC_${jobName}`) || !entry.endsWith('.safetensors')) continue;
    try {
      const ctime = fs.statSync(path.join(srcFolder, entry)).ctimeMs;
      if (ctime > criticCtime) {
        criticCtime = ctime;
        critic = entry;
      }
    } catch {
      // vanished mid-scan; skip it
    }
  }
  if (critic) {
    const dest = path.join(forkFolder, `CRITIC_${forkName}.safetensors`);
    fs.copyFileSync(path.join(srcFolder, critic), dest);
    copied.push(`${critic} -> CRITIC_${forkName}.safetensors`);
  }

  return copied;
};

export interface ForkJobOptions {
  // Test hook: override the name-uniqueness check (default: query the Job table).
  isNameTaken?: (name: string) => Promise<boolean>;
  // Test hook: override the training output root (default: the TRAINING_FOLDER setting).
  trainingRoot?: string;
}

// Build a fork of a training job: copy its current checkpoint + optimizer
// state into a fresh output folder under a new job name and return the new
// job's config. The caller inserts the DB row. The fork resumes automatically
// because the trainer loads the newest <job name>* file from its own save
// folder (weights + step/epoch from the safetensors training_info metadata +
// optimizer.pt), so no config change is needed besides the name.
export const forkJob = async (job: Job, options: ForkJobOptions = {}): Promise<ForkJobResult> => {
  if (job.job_type !== 'train') {
    throw new ForkJobError('Only training jobs can be forked', 400);
  }
  if (!FORKABLE_STATUSES.includes(job.status)) {
    throw new ForkJobError(`Cannot fork a job that is ${job.status}. Stop the job first, then fork it.`, 409);
  }

  const isNameTaken =
    options.isNameTaken ??
    (async (name: string) => (await prisma.job.findUnique({ where: { name } })) !== null);

  const trainingRoot = options.trainingRoot ?? (await getTrainingFolder());
  const srcFolder = path.join(trainingRoot, job.name);
  let srcStats: fs.Stats;
  try {
    srcStats = fs.statSync(srcFolder);
  } catch {
    throw new ForkJobError(`Job output folder not found: ${srcFolder}`, 404);
  }
  if (!srcStats.isDirectory()) {
    throw new ForkJobError(`Job output path is not a folder: ${srcFolder}`, 404);
  }

  const checkpoint = findLatestCheckpoint(srcFolder, job.name);
  if (!checkpoint) {
    throw new ForkJobError(
      'No checkpoint found. The job must save at least one checkpoint before it can be forked.',
      400,
    );
  }

  // Validate the config before touching the filesystem, so a bad row can't
  // leave an orphaned fork folder behind. Old job rows may predate parts of
  // the current config shape, so treat the parsed value as the domain type
  // but guard the fields we rewrite.
  let jobConfig: JobConfig;
  try {
    jobConfig = JSON.parse(job.job_config) as JobConfig;
  } catch {
    throw new ForkJobError('Job config is not valid JSON; cannot fork', 500);
  }
  if (!jobConfig?.config) {
    throw new ForkJobError('Job config is missing its config section; cannot fork', 500);
  }

  const forkName = await pickForkName(job.name, isNameTaken);
  const forkFolder = path.join(trainingRoot, forkName);
  fs.mkdirSync(forkFolder, { recursive: true });

  let copied: string[];
  try {
    copied = copyForkState(srcFolder, forkFolder, job.name, forkName, checkpoint);
  } catch (e) {
    // leave no half-copied fork folder behind
    try {
      fs.rmSync(forkFolder, { recursive: true, force: true });
    } catch {
      // best effort cleanup
    }
    throw e;
  }

  jobConfig.config.name = forkName;
  if (jobConfig.meta && typeof jobConfig.meta === 'object') {
    jobConfig.meta.name = forkName;
  }

  return {
    name: forkName,
    jobConfig,
    jobConfigJson: JSON.stringify(jobConfig),
    step: job.step,
    totalSteps: job.total_steps,
    copied,
  };
};
