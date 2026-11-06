import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { forkJob, ForkJobError } from '@/server/forkJob';

// Fork an in-progress training job: create a new job that resumes from the
// original's latest checkpoint (weights + optimizer state + step counter),
// so the original can continue while the fork explores different parameters.
export async function POST(_request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  let result;
  try {
    result = await forkJob(job);
  } catch (e) {
    if (e instanceof ForkJobError) {
      return NextResponse.json({ error: e.message }, { status: e.status });
    }
    console.error(e);
    return NextResponse.json({ error: 'Failed to fork job' }, { status: 500 });
  }

  try {
    const forkedJob = await prisma.job.create({
      data: {
        name: result.name,
        gpu_ids: job.gpu_ids,
        job_config: result.jobConfigJson,
        step: result.step,
        total_steps: result.totalSteps,
        status: 'stopped',
        info: `Forked from "${job.name}" (resumes at step ${result.step})`,
        job_type: job.job_type,
      },
    });

    console.log(`Forked job "${job.name}" -> "${result.name}": ${result.copied.join(', ')}`);
    return NextResponse.json(forkedJob, { status: 201 });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ error: 'Failed to create forked job' }, { status: 500 });
  }
}
