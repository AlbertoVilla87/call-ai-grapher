export interface AnalyzeResult {
  sessionId: string;
  charCount: number;
  labels: string[];
  before: string;
}

export interface RenderResult {
  after: string;
  elapsedMs: number;
}

export interface PipelineSettings {
  detectorBackend: "mser" | "yolo";
  yoloModel: string;
  confidence: number;
  classifierPath: string;
  stylizerBackend: "baseline" | "neural" | "latent";
  stylizerModel: string;
  autoencoderModel: string;
  alphabetDir: string;
  denoisePage: boolean;
}

export const DEFAULT_SETTINGS: PipelineSettings = {
  detectorBackend: "mser",
  yoloModel: "models/character_detector.pt",
  confidence: 0.25,
  classifierPath: "",
  stylizerBackend: "baseline",
  stylizerModel: "models/char_stylizer.pt",
  autoencoderModel: "models/char_autoencoder.pt",
  alphabetDir: "dataset/alphabet",
  denoisePage: false,
};

async function unwrap<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `The API answered ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* keep the status-line fallback */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export async function analyzePage(file: File, settings: PipelineSettings): Promise<AnalyzeResult> {
  const form = new FormData();
  form.append("image", file);
  form.append("detector_backend", settings.detectorBackend);
  form.append("yolo_model", settings.yoloModel);
  form.append("confidence", String(settings.confidence));
  form.append("classifier_path", settings.classifierPath);
  form.append("stylizer_backend", settings.stylizerBackend);
  form.append("stylizer_model", settings.stylizerModel);
  form.append("autoencoder_model", settings.autoencoderModel);
  form.append("alphabet_dir", settings.alphabetDir);
  form.append("denoise_page", String(settings.denoisePage));

  const data = await unwrap<AnalyzeResponseDTO>(
    await fetch("/api/sessions", { method: "POST", body: form }),
  );
  return {
    sessionId: data.session_id,
    charCount: data.char_count,
    labels: data.labels,
    before: data.before,
  };
}

interface AnalyzeResponseDTO {
  session_id: string;
  char_count: number;
  labels: string[];
  before: string;
}

interface RenderResponseDTO {
  after: string;
  elapsed_ms: number;
}

export async function renderPage(sessionId: string, alpha: number): Promise<RenderResult> {
  const data = await unwrap<RenderResponseDTO>(
    await fetch(`/api/sessions/${sessionId}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ alpha }),
    }),
  );
  return { after: data.after, elapsedMs: data.elapsed_ms };
}

export function releaseSession(sessionId: string): void {
  void fetch(`/api/sessions/${sessionId}`, { method: "DELETE", keepalive: true }).catch(() => undefined);
}
