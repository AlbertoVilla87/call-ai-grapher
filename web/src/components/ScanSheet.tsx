import { useId, useRef, useState } from "react";

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function useFilePicker(onFile: (file: File) => void) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const input = (
    <input
      ref={inputRef}
      id={inputId}
      type="file"
      accept="image/png,image/jpeg"
      hidden
      onChange={(event) => {
        const picked = event.target.files?.[0];
        if (picked) onFile(picked);
        event.target.value = "";
      }}
    />
  );
  return { input, pick: () => inputRef.current?.click() };
}

export function PageChip({ file, onFile }: { file: File; onFile: (file: File) => void }) {
  const { input, pick } = useFilePicker(onFile);
  return (
    <div className="page-chip">
      {input}
      <span className="page-chip-name" title={file.name}>
        {file.name}
      </span>
      <span className="page-chip-size">{formatSize(file.size)}</span>
      <button type="button" className="page-chip-action" onClick={pick}>
        Replace
      </button>
    </div>
  );
}

export function ScanHero({ onFile }: { onFile: (file: File) => void }) {
  const { input, pick } = useFilePicker(onFile);
  const [dragging, setDragging] = useState(false);

  return (
    <>
      {input}
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload a scanned page"
        className={`dropzone${dragging ? " is-dragging" : ""}`}
        onClick={pick}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            pick();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const dropped = event.dataTransfer.files?.[0];
          if (dropped && dropped.type.startsWith("image/")) onFile(dropped);
        }}
      >
        <span className="dropzone-glyph" aria-hidden="true">
          ✒
        </span>
        <p className="dropzone-title">Drop a scanned page here</p>
        <p className="dropzone-hint">or click to choose · PNG or JPEG</p>
      </div>
    </>
  );
}