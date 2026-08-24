import { useCallback, useEffect, useRef, useState } from "react";

const A4_PORTRAIT = 210 / 297;

export interface ComparePaneProps {
  before: string | null;
  after: string | null;
  rendering: boolean;
}

export default function ComparePane({ before, after, rendering }: ComparePaneProps) {
  const [split, setSplit] = useState(50);
  const [ratio, setRatio] = useState(A4_PORTRAIT);
  const paneRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  useEffect(() => {
    if (!before) return;
    const image = new Image();
    image.onload = () => {
      if (image.naturalWidth > 0) setRatio(image.naturalWidth / image.naturalHeight);
    };
    image.src = before;
  }, [before]);

  const updateFromPointer = useCallback((clientX: number) => {
    const pane = paneRef.current;
    if (!pane) return;
    const bounds = pane.getBoundingClientRect();
    const percent = ((clientX - bounds.left) / bounds.width) * 100;
    setSplit(Math.min(98, Math.max(2, percent)));
  }, []);

  if (!before) {
    return (
      <div className="compare-empty">
        <span className="compare-empty-glyph" aria-hidden="true">
          ✒
        </span>
        <p>The desk is clear.</p>
        <p className="hand">upload a page to begin.</p>
      </div>
    );
  }

  return (
    <figure
      ref={paneRef}
      className="compare"
      style={{ aspectRatio: String(ratio) }}
      onPointerDown={(event) => {
        draggingRef.current = true;
        event.currentTarget.setPointerCapture(event.pointerId);
        updateFromPointer(event.clientX);
      }}
      onPointerMove={(event) => {
        if (draggingRef.current) updateFromPointer(event.clientX);
      }}
      onPointerUp={() => {
        draggingRef.current = false;
      }}
      onPointerCancel={() => {
        draggingRef.current = false;
      }}
    >
      <img className="compare-img" src={before} alt="Original scanned page" />
      {after && (
        <img
          className={`compare-img compare-after${rendering ? " is-rendering" : ""}`}
          src={after}
          alt="Improved page"
          style={{ clipPath: `inset(0 0 0 ${split}%)` }}
        />
      )}
      <div
        className="compare-divider"
        style={{ left: `${split}%` }}
        role="slider"
        aria-label="Comparison position"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(split)}
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") setSplit((value) => Math.max(2, value - 3));
          if (event.key === "ArrowRight") setSplit((value) => Math.min(98, value + 3));
        }}
      >
        <span className="compare-nib" aria-hidden="true">
          ✒
        </span>
      </div>
      <span className="compare-tag compare-tag-before">before</span>
      <span className="compare-tag compare-tag-after">after</span>
    </figure>
  );
}
