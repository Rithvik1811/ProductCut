import type { NextRequest } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ jobId: string; ratioId: string }> }
) {
  const { jobId, ratioId } = await params;

  let backendRes: Response;
  try {
    backendRes = await fetch(`${BACKEND_URL}/jobs/${jobId}/download/${ratioId}`);
  } catch (err) {
    return new Response(`Backend unreachable: ${err}`, { status: 502 });
  }

  if (!backendRes.ok) {
    return new Response(`Backend error: ${backendRes.status}`, {
      status: backendRes.status,
    });
  }

  return new Response(backendRes.body, {
    status: 200,
    headers: {
      "Content-Type": "video/mp4",
      "Content-Disposition": `attachment; filename="${ratioId}.mp4"`,
      "Cache-Control": "no-store",
    },
  });
}
