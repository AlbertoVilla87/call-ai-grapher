import { useCallback, useEffect, useRef, useState } from "react";
import { analyzePage, DEFAULT_SETTINGS, releaseSession, renderPage, type Box, type PipelineSettings } from "./api";
import Header from "./components/Header";
import { PageChip, ScanHero } from "./components/ScanSheet";
import RefinePanel from "./components/RefinePanel";
import ComparePane from "./components/ComparePane";
import Ambient from "./components/Ambient";

interface Session {
  id: string;
  before: string;
  charCount: number;
  labels: string[];
  boxes: Box[];
}

function Kicker({ numeral, name }: { numeral: string; name: string }) {
  return (
    <p className="act-label">
      <span className="act-numeral display">{numeral}</span>
      <span className="act-name">{name}</span>
      <span className="act-rule" aria-hidden="true" />
    </p>
  );
}

const STATUS_MARK: Record<"quiet" | "busy" | "good" | "error", string> = {
  quiet: "✒",
  busy: "✒",
  good: "❧",
  error: "✕",
};

export default function App() {
  const [phase, setPhase] = useState<"idle" | "analyzing">("idle");
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [after, setAfter] = useState<string | null>(null);
  const [rendering, setRendering] = useState(false);
  const [alpha, setAlpha] = useState(0.8);
  const [settings, setSettings] = useState<PipelineSettings>(DEFAULT_SETTINGS);

  const renderTimer = useRef<number | undefined>(undefined);
  const renderTicket = useRef(0);

  useEffect(() => {
    return () => {
      window.clearTimeout(renderTimer.current);
    };
  }, []);

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
          return { id: result.sessionId, before: result.before, charCount: result.charCount, labels: result.labels, boxes: result.boxes };
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

  if (!file) {
    return (
      <>
        <Ambient />
        <div className="frame intro">
          <Header />
          <main className="intro-stage">
            <ScanHero onFile={handleFile} />
            <p className="intro-hint hand">your letter never leaves the desk</p>
          </main>
          <footer className="colophon rise">
            <span className="colophon-rule" aria-hidden="true" />
            <p>CallAIgrapher · MSER, YOLOv8, pix2pix &amp; latent blends inside</p>
          </footer>
        </div>
      </>
    );
  }

  return (
    <>
      <Ambient />
      <div className="frame working">
        <Header compact>
          <PageChip file={file} onFile={handleFile} />
          <p className={`strip-status status status-${status.tone}`} role="status">
            <span className="status-mark" aria-hidden="true">
              {STATUS_MARK[status.tone]}
            </span>
            <span className="status-text">{status.text}</span>
          </p>
        </Header>
        <main className="desk">
          <div className="desk-rail">
            <section className="sheet rise">
              <Kicker numeral="II" name="Refine" />
              <RefinePanel
                settings={settings}
                onSettings={setSettings}
                alpha={alpha}
                onAlpha={handleAlpha}
                hasPage={true}
                busy={phase === "analyzing"}
                onAnalyze={() => {
                  if (file) void runAnalyze(file);
                }}
              />
            </section>
          </div>
          <div className="desk-stage">
            <section className="sheet rise">
              <Kicker numeral="III" name="Compare" />
              <ComparePane before={session?.before ?? null} after={after} boxes={session?.boxes ?? []} rendering={rendering} />
            </section>
          </div>
        </main>
      </div>
    </>
  );
}