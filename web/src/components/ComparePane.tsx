import { useCallback, useEffect, useRef, useState } from "react";
import type { Box } from "../api";

const A4_PORTRAIT = 210 / 297;

export interface ComparePaneProps {
  before: string | null;
  after: string | null;
  boxes: Box[];
  rendering: boolean;
}

interface Size {
  width: number;
  height: number;
}

export default function ComparePane({ before, after, boxes, rendering }: ComparePaneProps) {
  const [split, setSplit] = useState(50);
  const [ratio, setRatio] = useState(A4_PORTRAIT);
  const [natural, setNatural] = useState<Size | null>(null);
  const [fit, setFit] = useState<Size | null>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const paneRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const draggingRef = useRef(false);

  useEffect(() => {
    if (!before) return;
    setNatural(null);
    const image = new Image();
    image.onload = () => {
      if (image.naturalWidth > 0) {
        setRatio(image.naturalWidth / image.naturalHeight);
        setNatural({ width: image.naturalWidth, height: image.naturalHeight });
      }
    };
    image.src = before;
  }, [before]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const measure = () => {
      const rect = stage.getBoundingClientRect();
      setFit({ width: rect.width, height: rect.height });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(stage);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const draw = () => {
      const canvas = canvasRef.current;
      const pane = paneRef.current;
      if (!canvas || !pane || !natural) return;
      const rect = pane.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, rect.width, rect.height);
      if (boxes.length === 0) return;
      const sx = rect.width / natural.width;
      const sy = rect.height / natural.height;
      ctx.strokeStyle = "rgba(196, 62, 40, 0.9)";
      ctx.lineWidth = 1.5;
      ctx.fillStyle = "rgba(196, 62, 40, 0.12)";
      for (const box of boxes) {
        const x = box.x * sx;
        const y = box.y * sy;
        const w = box.width * sx;
        const h = box.height * sy;
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);
      }
    };
    draw();
    const pane = paneRef.current;
    if (!pane) return;
    const observer = new ResizeObserver(draw);
    observer.observe(pane);
    return () => observer.disconnect();
  }, [boxes, natural]);

  const updateFromPointer = useCallback((clientX: number) => {
    const pane = paneRef.current;
    if (!pane) return;
    const bounds = pane.getBoundingClientRect();
    const percent = ((clientX - bounds.left) / bounds.width) * 100;
    setSplit(Math.min(98, Math.max(2, percent)));
  }, []);

  if (!before) {
    return <div ref={stageRef} className="compare-stage" aria-hidden="true" />;
  }

  const dims =
    fit && ratio > 0
      ? {
          width: Math.min(fit.width, fit.height * ratio),
          height: Math.min(fit.height, fit.width / ratio),
        }
      : null;

  return (
    <div ref={stageRef} className="compare-stage">
      {dims && (
        <figure
          ref={paneRef}
          className="compare"
          style={{ width: dims.width, height: dims.height }}
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
          <canvas ref={canvasRef} className="compare-boxes" aria-hidden="true" />
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
      )}
    </div>
  );
}