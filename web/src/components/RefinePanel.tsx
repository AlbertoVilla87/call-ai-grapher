import { useId } from "react";
import type { CSSProperties } from "react";
import type { PipelineSettings } from "../api";

export interface RefinePanelProps {
  settings: PipelineSettings;
  onSettings: (settings: PipelineSettings) => void;
  alpha: number;
  onAlpha: (alpha: number) => void;
  hasPage: boolean;
  busy: boolean;
  onAnalyze: () => void;
}

export default function RefinePanel({
  settings,
  onSettings,
  alpha,
  onAlpha,
  hasPage,
  busy,
  onAnalyze,
}: RefinePanelProps) {
  const confidenceId = useId();
  const patch = (changes: Partial<PipelineSettings>) => onSettings({ ...settings, ...changes });

  return (
    <>
      <div className="regulator">
        <div className="regulator-head">
          <span className="regulator-title">The regulator</span>
          <span className="regulator-value display">{Math.round(alpha * 100)}%</span>
        </div>
        <div className="regulator-track-row">
          <span className="hand regulator-end">your letter</span>
          <input
            className="ink-slider"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={alpha}
            aria-label="Improvement amount"
            style={{ "--fill": `${alpha * 100}%` } as CSSProperties}
            onChange={(event) => onAlpha(Number(event.target.value))}
          />
          <span className="hand regulator-end">its best self</span>
        </div>
      </div>

      <details className="drawer">
        <summary>Quills &amp; inks — pipeline settings</summary>
        <p className="drawer-note">Model paths are resolved on the machine running the API.</p>

        <fieldset className="field">
          <legend>Detector</legend>
          <div className="pill-row" role="radiogroup" aria-label="Detector backend">
            {(["mser", "yolo"] as const).map((value) => (
              <label key={value} className={`pill${settings.detectorBackend === value ? " is-on" : ""}`}>
                <input
                  type="radio"
                  name="detector"
                  value={value}
                  checked={settings.detectorBackend === value}
                  onChange={() => patch({ detectorBackend: value })}
                />
                {value}
              </label>
            ))}
          </div>
          {settings.detectorBackend === "yolo" && (
            <>
              <label className="ledger">
                <span>YOLO weights</span>
                <input
                  type="text"
                  value={settings.yoloModel}
                  onChange={(event) => patch({ yoloModel: event.target.value })}
                />
              </label>
              <label className="ledger ledger-range">
                <span>
                  Confidence <b>{settings.confidence.toFixed(2)}</b>
                </span>
                <input
                  id={confidenceId}
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={settings.confidence}
                  onChange={(event) => patch({ confidence: Number(event.target.value) })}
                />
              </label>
            </>
          )}
        </fieldset>

        <fieldset className="field">
          <legend>Stylizer</legend>
          <div className="pill-row" role="radiogroup" aria-label="Stylizer backend">
            {(["baseline", "neural", "latent"] as const).map((value) => (
              <label key={value} className={`pill${settings.stylizerBackend === value ? " is-on" : ""}`}>
                <input
                  type="radio"
                  name="stylizer"
                  value={value}
                  checked={settings.stylizerBackend === value}
                  onChange={() => patch({ stylizerBackend: value })}
                />
                {value}
              </label>
            ))}
          </div>
          {settings.stylizerBackend === "neural" && (
            <label className="ledger">
              <span>pix2pix checkpoint</span>
              <input
                type="text"
                value={settings.stylizerModel}
                onChange={(event) => patch({ stylizerModel: event.target.value })}
              />
            </label>
          )}
          {settings.stylizerBackend === "latent" && (
            <>
              <label className="ledger">
                <span>Autoencoder checkpoint</span>
                <input
                  type="text"
                  value={settings.autoencoderModel}
                  onChange={(event) => patch({ autoencoderModel: event.target.value })}
                />
              </label>
              <label className="ledger">
                <span>Alphabet dataset</span>
                <input
                  type="text"
                  value={settings.alphabetDir}
                  onChange={(event) => patch({ alphabetDir: event.target.value })}
                />
              </label>
              <label className="ledger">
                <span>Classifier checkpoint · required</span>
                <input
                  type="text"
                  value={settings.classifierPath}
                  placeholder="models/char_classifier.pt"
                  onChange={(event) => patch({ classifierPath: event.target.value })}
                />
              </label>
            </>
          )}
        </fieldset>

        <label className="toggle">
          <input
            type="checkbox"
            checked={settings.denoisePage}
            onChange={(event) => patch({ denoisePage: event.target.checked })}
          />
          <span className="toggle-ui" aria-hidden="true" />
          Flatten shadows &amp; specks before detection
          <em>recommended for phone photos</em>
        </label>
      </details>

      <button
        type="button"
        className={`ink-btn${busy ? " is-busy" : ""}`}
        onClick={onAnalyze}
        disabled={busy}
        title={hasPage ? "Run detection and refinement over the page" : "Choose a scanned page"}
      >
        <span className="ink-dot" aria-hidden="true" />
        <span className="ink-label">{busy ? "The scribe is working…" : hasPage ? "Refine the page" : "Choose a page"}</span>
      </button>
    </>
  );
}
