import { getMachineDescription } from "@/lib/server/episode-runs";

export async function GET() {
  return Response.json(await getMachineDescription());
}
