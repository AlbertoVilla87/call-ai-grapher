import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { analyzePage, DEFAULT_SETTINGS, releaseSession, renderPage, type PipelineSettings } from "./api";
import Header from "./components/Header";
import ScanSheet from "./components/ScanSheet";
import RefinePanel from "./components/RefinePanel";
import ComparePane from "./components/ComparePane";
import Ambient from "./components/Ambient";

interface Session {
  id: string;
  before: string;
  charCount: number;
  labels: string[];
}

function Act({ numeral, name, children }: { numeral: string; name: string; children: ReactNode }) {
  return (
    <section className="sheet rise">
      <p className="act-label">
        <span className="act-numeral display">{numeral}</span>
        <span className="act-name">{name}</span>
        <span className="act-rule" aria-hidden="true" />
      </p>
      {children}
    </section>
  );
}

export default function App() {
  const [phase, setPhase] = useState<"idle" | "analyzing">("idle");
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [after, setAfter] = useState<string | null>(null);
  const [rendering, setRendering] = useState(false);
  const [alpha, setAlpha] = useState(0.8);
  const [settings, setSettings] = useState<PipelineSettings>(DEFAULT_SETTINGS);

  const renderTimer = useRef<number | undefined>(undefined);
  const renderTicket = useRef(0);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      window.clearTimeout(renderTimer.current);
    };
  }, [previewUrl]);

  const runRender = useCallback(async (sessionId: string, amount: number) => {
    const ticket = ++renderTicket.current;
    setRendering(true);
    try {
      const result = await renderPage(sessionId, amount);
      if (ticket === renderTicket.current) setAfter(result.after);
    } catch (cause) {
      if (ticket === renderTicket.current) setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (ticket === renderTicket.current) setRendering(false);
    }
  }, []);

  const runAnalyze = useCallback(
    async (source: File) => {
      window.clearTimeout(renderTimer.current);
      setPhase("analyzing");
      setError(null);
      try {
        const result = await analyzePage(source, settings);
        setSession((previous) => {
          if (previous) releaseSession(previous.id);
          return { id: result.sessionId, before: result.before, charCount: result.charCount, labels: result.labels };
        });
        setPhase("idle");
        void runRender(result.sessionId, alpha);
      } catch (cause) {
        setPhase("idle");
        setError(cause instanceof Error ? cause.message : String(cause));
      }
    },
    [settings, alpha, runRender],
  );

  const handleFile = useCallback(
    (picked: File) => {
      setFile(picked);
      setPreviewUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return URL.createObjectURL(picked);
      });
      void runAnalyze(picked);
    },
    [runAnalyze],
  );

  const handleAlpha = useCallback(
    (value: number) => {
      setAlpha(value);
      if (!session) return;
      window.clearTimeout(renderTimer.current);
      renderTimer.current = window.setTimeout(() => void runRender(session.id, value), 250);
    },
    [session, runRender],
  );

  let status = { tone: "quiet" as "quiet" | "busy" | "good" | "error", text: "Upload a scanned page to begin." };
  if (error) status = { tone: "error", text: error };
  else if (phase === "analyzing") status = { tone: "busy", text: "Reading the page…" };
  else if (rendering && session) status = { tone: "busy", text: "The scribe rewrites…" };
  else if (session) {
    const found = `${session.charCount} letters found`;
    const labels = session.labels.slice(0, 12).join(", ");
    status = { tone: "good", text: labels ? `${found} · ${labels}` : `${found} · drag the nib to compare` };
  }

  return (
    <>
      <Ambient />
      <div className="frame">
        <Header />
        <main className="desk">
          <div className="desk-rail">
            <Act numeral="I" name="Scan">
              <ScanSheet file={file} previewUrl={previewUrl} onFile={handleFile} />
            </Act>
            <Act numeral="II" name="Refine">
              <RefinePanel
                settings={settings}
                onSettings={setSettings}
                alpha={alpha}
                onAlpha={handleAlpha}
                hasPage={file !== null}
                busy={phase === "analyzing"}
                onAnalyze={() => {
                  if (file) void runAnalyze(file);
                }}
              />
            </Act>
          </div>
          <div className="desk-stage">
            <Act numeral="III" name="Compare">
              <ComparePane before={session?.before ?? null} after={after} rendering={rendering} />
              <p className={`status status-${status.tone}`}>
                <span className="status-mark" aria-hidden="true">
                  {status.tone === "error" ? "✕" : status.tone === "good" ? "❧" : "✒"}
                </span>
                {status.text}
              </p>
            </Act>
          </div>
        </main>
        <footer className="colophon rise">
          <span className="colophon-rule" aria-hidden="true" />
          <p>CallAIgrapher · MSER, YOLOv8, pix2pix &amp; latent blends inside · your letter never leaves the desk</p>
        </footer>
      </div>
    </>
  );
}
